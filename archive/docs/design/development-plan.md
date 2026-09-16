# QSA HiSparse 下一阶段开发计划

日期：2026-09-09。本轮仅规划，不修改候选源码、不运行 GPU、不改变基线服务。

## 目标与边界

实现“一条请求占用 prefill 工作区，多条请求共同 decode”的可测服务。先解决单请求交接与 decode 开销，再建立多请求所有权，最后逐级验证 B2/B4/B8。B8×近256K 是待测目标，不是已有容量或性能承诺。

同一时刻最多一个 prefill owner，其他等待 prefill 的请求排队。允许唯一 prefill 请求的 chunk 与已有 decode 交替调度；不要求二者在 GPU 上同时执行。最大运行数为8时，在线场景是7条 decode＋1条 prefill，随后8条 decode，不隐含第9条请求的预算。

暂不做多请求同时 prefill、prefill 历史 KV 的 CPU 流式加载、PD、radix/prefix sharing、MTP/speculation 或完整上游合并。沿用现有 QSA 数学路径、FP8 writer、C4 表示和已验证 indexed unpack；只有 profile 证明有必要才新增内核。

## 已有基线

- 候选源码：`f04c0d795509c1c52df0147a893a35b634ad20a0`；生产：`bd83f02`。报告与原始证据见同目录 `ANALYSIS-RESULTS.md`、`EXECUTION-02-METRICS.json`。
- 已接受 `PASS_V3_SINGLE_REQUEST_EAGER`：261120输入、768输出，TP2每rank 12Q/1KV/head_dim256，12层；resident/offload输出完全一致，尾部、取消、释放、同进程再准入通过。
- 验证运行 resident/offload TTFT约43.16/43.22秒，平均TPOT约94.15/111.93毫秒。offload首次decode间隔5.62秒；去掉首次间隔后的均值约104.75毫秒。仍包含字节oracle及事件日志，不能作为生产性能基线。
- 单请求交接实降allocated约1.439 GiB/rank，但原型限定单owner、共享workspace，并整体替换raw池。现有logical/index容量也只覆盖262144个token；调大max_running_requests并不能获得多并发。

## 顺序、产物与晋级条件

### P0：分离正确性校验与轻量性能基线

最小改动：保留当前重型校验模式，增加明确的轻量运行路径。轻量路径关闭逐步字节回读、pointer扫描和逐事件文件写入；保留边界检查、请求身份检查、必要stream/event依赖及错误处理。不可用“关闭校验”隐藏依赖错误。

冻结同源、同输入、同seed、同后端和相同调度选项的resident/offload对照。先用原A请求作3组成对运行，交替次序，分别报告每次结果和漂移；重复baseline兼作A/A噪声估计。工程筛选要求median收益超过2倍baseline MAD，并超过前后baseline漂移；样本不足以分辨时不报收益，再补必要重复。这是筛选规则，不等同统计显著性证明。正确性回归使用完整768输出，性能主运行不同时开启重型profile或oracle。现有生产配置可单列参考，不能把graph、MTP等配置差异算成offload收益。

必须拆开：TTFT、prefill吞吐、handoff wall、首次输出间隔、steady inter-token P50/P95/P99、请求e2e、aggregate输出tokens/s，以及双rank中较慢者的关键路径。handoff内拆gather/pack、D2H、CPU等待；decode内拆QSA index、miss/refetch、unpack、FA2与CPU发射。CUDA事件总和不能代替端到端wall。

晋级：轻量模式完整输出与已验收基线一致；明确112毫秒中校验、交接、steady路径分别占多少。以这批结果冻结后续性能比较合同。未预先定义延迟 SLO 时，不宣称生产性能 GO。

### P1：先消除B1的主要交接和decode开销

根据P0关键路径按收益排序，一次改变一类因素：

1. 交接复用固定packing buffer和事件，在容量有界的流水线内重叠packing与D2H；只在真正消费或复用buffer前等待。保留完整host ready和失败回收协议，避免每个chunk立即CPU同步。
2. 删除轻量路径不必要的`.cpu()`/`.item()`同步、临时tensor构造和重复分配，复用compact K/V、索引及事件；不重写无证据表明昂贵的上游路径。
3. 保持已验证单次indexed unpack，测真实decode消费后的收益。如果后续compact写入与FA2再次extraction占据可见开销，优先直接接入已有packed K/V消费入口以去掉重复整理；先核对scale、tail与布局合同，不立即写新融合内核。保留槽0与C4 member 1..4约定、newest保护及未完成tail。

晋级：原长请求输出、tail0..3、跨page边界及真实miss/refetch正确；取消和再准入不退化。改动的端到端收益须超过成对基线漂移，且不能用更坏的steady P95换取只改善平均数。没有可信收益的改动不保留。历史V4的2.427575毫秒只作组件参考，不作为服务TPOT门槛。

### P2：解耦容量与所有权，先打通B2 eager

实现优先复用现有 HiSparse coordinator 的 host-ready 准入、双 rank 事件收敛和 allocator 逻辑/物理分离做法；MLA 专用数据布局不直接搬进 QSA，也不新增通用 offload 框架。

改动集中于现有adapter、QSA pool及其必要调用点：

- **长期logical/index空间**：承载所有活动请求的index-K和请求映射，直到请求终止才归还。扩容它不能连带分配同等容量的full raw KV。
- **单一prefill physical staging**：容量只覆盖一条最大上下文，加必要padding。保留稳定pool对象和地址，交接后由下一条prefill复用；停止每次全局替换/重建raw池。prefill的逻辑token须映射到其physical staging，不能再假设两者地址相同。
- **每请求decode状态**：以request slot＋generation识别，隔离host KV、hot映射、C4 tail/ring、newest写回及终止事件。不同请求的同一logical block不能互相命中，slot复用不能收到旧事件。
- **batch workspace**：按已支持batch容量预分配，由当前执行batch独占；batch行与request slot显式映射。生命周期不重叠时可共享，无需为每请求复制全部scratch。
- **准入与释放**：prefill完成后，双rank host ready才进入decode；staging的全部使用者完成后才转交新prefill。取消须drain后释放、postflush后归还logical lease。一个请求释放不能改变其他请求的index-K/Mamba/host/hot。

本阶段先允许新prefill期间暂停已有decode，以最小调度改动验证容量和隔离；这一状态只授予功能通过，不授予在线服务可用性。

晋级：两条不同近256K请求真实共同decode；相同block编号、不同payload、错开tail、A终止/B继续、slot复用、取消后再准入均通过。raw staging只分配一份；full pool不随logical容量线性增大；实际free-list、backing与host生命周期闭环。

### P3：批量decode与CUDA graph

在P2稳定地址及所有权基础上，先实现正确的batch元数据和indexed unpack，测B1/B2 eager；随后接入现有CUDA graph机制，按实际支持的batch size捕获。host事件处理、请求准入和释放留在graph外。动态miss数量应通过固定容量buffer和有效长度处理，不能为捕获而删掉必需的H2D/D2H依赖。

逐项检查request重排、batch缩小/扩大、graph padding槽0、C4 close与旧请求slot复用；多请求C4余数不同，close/tail必须按row判断，不能沿用单请求Python四相位分支。明确graph replay完成前哪些host/GPU buffer不得改写或释放。若完整路径不可捕获，先记录具体阻碍与可测收益，接受局部eager的阶段结果，不声称完整graph支持。

晋级：相同batch/config下eager与graph输出及独立K/V检查通过，连续运行无地址或生命周期错误，steady延迟/吞吐改善超过漂移。graph pool与预分配scratch计入显存；先B1/B2，再随P5放开B4/B8，不一次捕获所有无需求尺寸。

### P4：唯一prefill与已有decode交替调度

复用现有chunked-prefill和调度入口，每个调度周期最多推进同一条prefill的一个受预算约束chunk，并给已有decode运行机会；按源码实际支持选择交替batch或兼容的混合batch，不要求立即开启通用overlap scheduler。

验收B1 decode＋1条新prefill开始，随后扩大到B4/B8允许的运行数。分别记录已有请求最大输出间隔、steady P95、incoming TTFT和aggregate吞吐。prefill工作区只准入一个owner；新请求不能覆盖现有decode的数据。若chunk或handoff仍造成长停顿，按测量减小chunk或拆分handoff调度；不能只看8条都最终完成。

晋级：功能、事件依赖与资源隔离通过，已有decode在prefill期间持续取得进展。无延迟SLO时提供最大停顿实测和吞吐权衡，只报混合负载测量通过；上线资格须绑定明确可接受的TTFT/TPOT/最大停顿预算。

### P5：B1→B2→B4→B8逐级真实服务验收

前一级正确性或容量失败则停止放大。分两个场景：

1. **共同decode容量**：顺序完成prefill并保留活动请求，再形成真实B条decode；验证服务端每步有效batch及请求身份，不能用B个排队客户端代替。必要的测试屏障只用于形成这一负载，不能据此计算在线TTFT。
2. **在线到达**：B−1 decode期间到达一条新prefill，随后B条共同decode；测TTFT、输出间隔、吞吐和全阶段资源峰值。

每一级包含异步完成、取消和同 slot 重新准入；已有 tail/refetch/释放检查复用，避免每个优化重跑全部诊断矩阵。最终候选再做原 768 输出回归与双 rank 生命周期复核。

正确性对照：B1沿用完整greedy金标；多请求优先与相同B、相同调度的resident对照。若近256K resident同B放不下，使用可容纳上下文的同B对照，加长上下文固定token轨迹的独立host K/V oracle、单请求金标定位。跨batch输出差异须区分数值路径与缓存错误，不能默许放宽容差；证据不足时明确标记多请求输出验收未完成。

结果分别裁决correctness、capacity和performance。报告每请求及aggregate指标；B8容量通过不等于B8延迟可接受，更不等于8倍吞吐。

## 容量预算与停止条件

以下为当前表示推算的每rank payload，不是实测总峰值；256K按262144 token，12层计算。

| 活动请求B | pinned host KV | index-K payload | GPU hot（当前布局） | 最少logical token容量 |
|---:|---:|---:|---:|---:|
| 1 | 1.5 GiB | 0.1875 GiB | 49.5 MiB | 262144 |
| 2 | 3 GiB | 0.375 GiB | 99 MiB | 524288 |
| 4 | 6 GiB | 0.75 GiB | 198 MiB | 1048576 |
| 8 | 12 GiB | 1.5 GiB | 396 MiB | 2097152 |

另外预留一份约1.5 GiB/rank的raw prefill staging、每请求30 KiB的tail ring、batch workspace、请求/逻辑映射、padding、graph pool及prefill临时tensor。保留原权重和Mamba常驻开销；当前Mamba池约2.204 GiB/rank是预分配池，不能按B重复乘算。

B8双rank host KV合计约24 GiB，需核对两张卡相应NUMA节点的可用内存与pinning额度。显存按每卡独立计算，不能将两卡剩余显存相加。新的稳定staging设计会保留这份显存供后续prefill，因此不能照搬旧“交接即减少1.439 GiB”的单请求观测作为新稳态预算。

每次扩容前先算账，再测CUDA allocated/reserved、采样NVML峰值、host pinned和free-list；必须涵盖prefill、交接、graph capture和请求更替，注明采样无法证明连续时间绝对峰值。预算不足、OOM、跨请求污染或释放不完整即停在前一级，不为了B8临时缩短上下文后沿用B8×256K名称。

## 执行安排与版本策略

- 接口调研确认先拆校验、解耦 staging 和请求状态，再做 graph。B2 eager 先证明所有权，随后接 graph 并扩大 batch；B4/B8 各自补齐 eager 正确性对照。
- 小改动复用已有检查，只为新风险补最少且有意义的测试。
- 长任务使用阻塞 supervisor，结束后发出一次完成事件，不反复轮询。
- 每阶段固定 source commit 和配置，保存逐请求 JSONL、服务日志及分项判定；CPU 检查和审查通过后才进入 GPU 窗口。
- 基线服务保持原 source；需要 GPU 窗口时先验证 3 秒无 decode 再停机，candidate 使用独立测试端口，退出时恢复并核验基线配置。
- 不先合并上游。只做差异阅读和接口复用；若发现必须修复的问题，隔离最小cherry-pick或移植，并重新绑定基线。整体升级单独立项，避免混淆收益归因。

**下一项具体工作：P0轻量模式与B1成对基线；P1只处理测出的主要瓶颈，随后进入P2的logical/index与physical staging解耦。**
