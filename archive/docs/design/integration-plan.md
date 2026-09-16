# QSA–HiSparse 接入原型验证计划

本计划冻结 QSA–HiSparse 原型的验证范围、顺序和通过条件。

## 目标与固定基线

回答真实 QSA 热缓存布局、生成 KV 生命周期、同机 prefill 交接和多请求搬运是否足以支持正式服务接入设计。允许编写最小隔离原型、必要观测与测试；不把任务扩大为完整生产 offload 部署。

- 已有证据：`../qwen38-hisparse-tests-20260907/ANALYSIS-PROTOCOL.md`、`ANALYSIS-RESULTS.md` 和 execution-02 两份独立审查。旧阶段 E0–E3 保留，不重复全矩阵。
- 源码固定 `d0664112a8c81dd8019689bc2dd5e5b8fc4f851a`。从该版本另建 worktree/分支实施原型；不得改运行中的 `sources/sglang`，不得改旧 execution-02 文件。
- 上游已 fetch 到 `afe90a8bc908002219993794404c7c95dd3ced1d`；本轮不 merge/rebase、不升级运行依赖。源代码和验证脚本均以实际 Git commit 绑定测量。
- 测试配置使用 Qwen3.8-Flash-Next W4A16 RTN AutoRound INT8PLE，TP2、no-spec、FP8 KV、FP32 GDN、BF16 index、page64、C4、top512 C4、12 QSA 层。缓存4×=2048 C4/层/请求；热 KV48MiB/请求/卡，不等于每步搬48MiB。
- 真实长输入复用旧 request-A/B：261120输入、768输出。服务基线和请求输出ID均可复用；同一数值比较必须保持模型、输入、selection、scale、mask和求和顺序契约一致。

## 执行顺序与最小矩阵

### V0：接口与参考冻结（CPU，先做）

沿实际 QSA pool→allocator→位置展开→attention 调用追踪，给出分离K/V、packed C4、index-K、尾部和逻辑/物理映射的 shape、dtype、stride、字节账本。明确通用HiSparse allocator可复用部分与必需adapter，不把DSV4类型契约直接套给QSA。

复用已有CPU参考/测试。候选运行前，把原型方案、测试输入、独立参考来源和验收断言写到本目录的 CONTRACT.md 并提交。未解决的语义歧义阻塞相关实现，其他独立阶段可继续；不得从候选结果反推容差。

### V1：真实 QSA 消费热缓存（P0）

使用实际QSA attention入口验证 resident KV 与 host→hot KV 两种来源。覆盖真实A/B trace的首/中/末步、12层、双rank；覆盖C4余数0/1/2/3、page64边界、非单位scale、有效尾部和合法mask。先固定同一Q、selection和token顺序，保证只有数据存储位置不同。

要求KV round-trip逐字节相等；同一attention backend、运算顺序的输出要求逐位一致。独立FP32参考沿用既有已建立容差；若无适用容差，先从独立误差来源确立合同，不以候选误差调松。记录实际attention消费、映射、必要gather的CUDA Graph成本和workspace，不能只测复制后generic gather就宣称完成V1。

最小故障检查：故意破坏一个物理映射或scale，验证检查确实拒绝。测量之外复用一次小自检，不堆测试框架。

### V2：C4生成与写回生命周期（P0）

最小adapter执行 append→C4闭组→D2H完成→允许驱逐→H2D重取→attention消费；明确未闭组0–3个原token和最新完整C4各自归属。强制构造hot capacity耗尽后的驱逐/重取，并覆盖请求释放、slot重新准入及尚未完成copy时的释放顺序。

两rank均检查字节、scale、映射唯一性、selected set完整驻留、无读取未完成写回、无跨请求污染。至少一个故意提前复用slot/漏掉依赖event的反例须被检查拒绝。记录host/device tensor实际字节、pinned host总量、元数据更新及同步的wall time/CUDA event时间；计入旧E3未测的成本。

V1/V2正确性失败即停止依赖性能结论；保留失败attempt，可在独立修复commit后重跑受影响项。禁止靠全局每步device synchronize隐藏原拟异步路径的竞态；若首版选择同步实现，明确其契约并计入实际等待。

### V3：同机 prefill→decode 交接（P1）

先用一条A-long、现有chunk2048确认真实模型KV交接，不扫描chunk参数。测 cold prefill、完整GPU KV→host/hot交接、交接后首个decode的耗时及峰值显存/锁页内存；记录临时buffer、全驻留pool引用和实际释放时点。

必须区分allocator allocated/reserved峰值与NVML进程占用；端点采样不能称峰值。若当前模型仍持有完整GPU pool，应如实记为未回收，不能以删除临时tensor或退出模型进程冒充服务释放。再用B-long复核仅当不同内容会影响被验证路径，或A暴露新问题；本阶段目的不是再做完整基线测量。

若真实交接只能通过完整服务allocator改造实现，交付定位到具体接口的 BLOCKED/SERVICE_DEPENDENCY 和实测峰值，不把合成搬运冒充真实交接，也不擅自扩展为生产接入。

### V4：1/4/8独立请求的组件争用（P1）

每请求独立host/hot pool、逻辑地址映射和生成状态；使用A/B真实trace错开起点构造请求，不利用共同前缀共享缓存。双卡同时、各12层串行，测试B=1/4/8；H2D字节由GPU真实miss计划决定。逻辑总容量分别按262144/1048576/2097152原token配置，不能仍用单请求总池冒充8个独立长请求。

每档3次独立重放。记录每步双rank较慢值、各请求延迟、整批完成时间、搬运吞吐、P50/P95/P99、实际D2H/H2D字节、host NUMA观察和GPU/host峰值。V2写回在路径可用时必须开启；未实现则明确保留只读范围。冷L2只复用既有128MiB机制，不模拟完整模型权重争用。

B=1按同一trace的已有无observer TPOT×10%作为组件筛选预算，计量范围必须明确；B=4/8报告扩展趋势，不拿B=1的24ms当成未测的多请求服务基线，不据此给出8并发服务GO。

## 生命周期、版本与运行方式

复用旧 run_tests.py 中已验证的快照、idle gate、launch、stop 和 finally restore helpers；只加本轮阶段需要的调度。先完成 CPU 准备和必要编译，再申请独占 GPU 窗口。基线停机前检查实时 running/queue=0、日志连续 3 秒无 decode/prefill，测试端口及 GPU 无其他实验所有者。忙时不抢占，保留可续跑状态。

快照基线启动参数和必要环境，敏感环境仅留进程内存。任何成功、失败或异常路径都恢复并验证相同基线，恢复失败单列 BLOCKED/RESTORE。不得安装或更改部署依赖。

长 GPU 任务使用持久 driver、每阶段增量 manifest、独立 attempt 目录；用进程退出事件发送一次完成或失败通知，不反复轮询。

所有时间统一记录为 UTC。结果、逐请求/逐步 JSONL、日志窗口、失败记录及版本放本目录，禁止文件哈希计算。仅提交自身文件，不纳入其他实验的未提交改动。

## 交付和裁决

执行者交付 CONTRACT.md、最小原型/检查命令、各阶段manifest和数据、ANALYSIS-RESULTS.md、恢复验证、实际源码及脚本commit。逐项PASS/FAIL/BLOCKED，明确未测内容与能否进入正式服务接入。

最高结论是接入设计已有证据支持；本轮仍不能替代完整模型768步resident/offload输出比较、真实多请求cold-prefill准入、服务TPOT/吞吐和异常恢复验收。最终结论必须经过结果复核。
