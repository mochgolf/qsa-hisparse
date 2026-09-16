# QSA–HiSparse 接入前完整测试协议

本协议在新的长上下文测量及 GPU 搬运结果产生前冻结。

本轮交付接入决策所需的完整证据；standalone 微基准不能冒充端到端服务部署。

## 固定合同与环境

- 基线来源：`../qwen38-hisparse-assessment-20260907/ANALYSIS-HISPARSE-DEPLOYMENT.md`；旧38K第四窗口，源码`6f25e04479152cfe68e76a16b5c00236a6f360f8`。
- 隔离源码：`<SGLANG_WORKTREE>`，分支`codex/qsa-hisparse-tests-20260907`。运行前提交变更，以实际git commit绑定测试；禁止修改共享运行源码、模型权重。
- 模型：未消融W4A16 RTN AutoRound INT8PLE；TP2、no-spec、FP8 E4M3 KV、FP32 GDN、page64、QSA C4/top512、NUMA3/2、确定性FlashInfer、full decode graph/prefill graph disabled。context及旧总token池均262144。
- 数学与布局：12个QSA层；每卡每层每原token的K+V为512B，每C4 raw-KV为2048B。BF16 index-K每4token一条128维，驻留GPU。热缓存不改变selection、KV字节、scale、mask或位置语义。
- 已知旧trace不是256K，不以重复填充扩大输入；两条long使用不同真实内容，实际input_ids目标261120，输出768，合计不超过262144。
- 本轮首先测试统一4×缓存：每层每请求2048 C4 blocks = 8192 raw token slots，top512 C4；每层热KV4MiB、12层合计48MiB/请求/卡。manifest同时记录block数、raw token数和实际字节，防止将2048误传给raw-token allocator。同等字节等于128个page64页的容量，但不表示这128页能容纳离散selected page set。同时CPU回放1×/2×/3×/4×/8×用于判断收益与容量。

## 资源与恢复约束

登记时另一消融实验正在使用测试端口和双 GPU。不得中断、混跑或修改其 resident；GPU 阶段等待该实验真正退出并完成恢复。CPU 输入、回放、编译和审查可先做。

GPU 窗口开始时重新核验进程与源码，快照基线命令和必要运行环境；确认日志连续 3 秒无 decode/prefill 且没有其他实验 driver，再临时停止服务。模型实验仅用独立测试端口。任何失败均清理本任务子进程，并恢复窗口开始时捕获的基线配置。不得将新实验 candidate 留作默认部署。

## 测试矩阵与执行次序

| 阶段 | 输入/执行 | 必须保存 | 通过标准 |
|---|---|---|---|
| E0 CPU准备 | 旧38K数据；A-long真实代码、B-long真实技术文档 | 来源清单、任务文本、实际input_ids、请求JSON；泛化回放脚本 | 无重复填充；输入长度真实；旧1×miss精确等于1579601/4706304；synthetic策略/保护检查通过 |
| E1 无observer基线 | 旧38K与A/B-long，每个1次warmup+2次测量，单请求 | 逐请求JSONL、SSE计时、input/output IDs、日志窗口、源码及启动参数、内存 | 生成完成且无OOM/错误；实际768输出；正常full graph decode；TTFT与decode分列，不以SSE事件数冒充token数 |
| E2 长轨迹及top-k哨兵 | A/B-long各一次768输出，原r1 observer，两个TP rank | 每步每层原始C4集合；每层8个真实logits行及对应selection；generation IDs | 每rank12×767行、position连续、512唯一合法ID；两rank逐key比较；每个sentinel满足合法top512（边界并列允许） |
| E2R CPU完整回放 | 三条trace、五档容量，按请求/层隔离 | 全程/前128步/其余窗口；逐层和同token12层sum；miss IDs与需求字节 | selected set始终完整驻留；容量不越界；旧恒等式；首装、历史miss、生成后新block需求独立，未完成尾部不冒算成H2D |
| E3 GPU正确性与成本 | 三条真实地址trace，首测4×；双卡同时、每卡12层串行、CUDA Graph | resident读取A、GPUresolve无IO B、GPUresolve+真实pinned-host读取C；逐步events/内存/hostNUMA | resolve由GPU实际决定hit/miss/victim；fetch字节/scale正确；cache和所选集合不变量通过；所有错误保留并停止对应性能结论 |
| E3D 决策 | 各trace对应的E1基线和E3结果 | 净显存账本、延迟分位数、GO/NO-GO/容量用途待评估 | 见下文决策门槛；不把微基准当端到端资格 |

E1若observer必须引入额外graph copy，须使用关闭该功能的独立启动窗口；仅停止CPU drain并不能移除捕获的copy节点。E2优先沿用producer引用与post-graph drain，避免改变选择路径。

Sentinel采样decode步序号固定为0/1/2/3/127/255/511/766，覆盖开始、结束及C4边界。CPU参考检查有效候选域的logits，要求selected的最小分数不小于未选中的最大分数；不以候选结果调松容差，不把越界、NaN或不合法集合过滤掉。该哨兵验证top-k选择，不声称验证模型回答质量。

CPU回放策略固定：同一步先对完整selection判断hit/miss，保护全部选中项，再驱逐未选中项；hit较新miss优先成为MRU；组内按block ID确定tie。64-token整页方案先检查相同显存预算能否容纳完整selected page set，不能驱逐当前所需页来伪造成功。已知38K五档均存在整页容量不可行步骤。

E3必须使用真实trace的长地址跨度。可用可复现的有限FP8 payload验证搬运机制，但须明确其不是模型真实KV/输出质量验证；host若采用C4 packed布局，须验证native分离K/V到packed再到gather的字节一致及非单位scale保留。B省略IO产生的数据不能用于正确性或模型输出，只能作resolve开销对照。CPU oracle只能验算，不能预先替GPU决定C的miss计划。

E3重放顺序固定A/B/C；每条trace每变体做3次独立完整重放，每次从相同首selection初始化缓存及allocator，不沿用上次结束状态。编译和graph warmup在正式重放外完成，其后重置状态；前128步/其余步单列。统计须先合计同token的12层，双卡分别报告并提供较慢rank步时延，不相加两卡时延或各层P99。host first-touch分别绑定GPU邻近NUMA节点，记录实际放置。新KV写回未实现时必须单列遗漏，不能称完整生命周期成本。C−A为相同trace/step/重复序号的组件差值，不是服务端到端TPOT改善。

L2敏感性：standalone热cache合计48MiB/卡可能落入4090的L2，实际模型权重会冲刷它。在正式GPU前固定增加cold-cache对照：每步计时前触及至少128MiB独立eviction buffer，A/B/C同样处理，eviction不计入events。可全部trace执行；若为节约只测一条，选择规则固定为主实验C−A P95最大的trace，不能挑最有利结果。报告它只近似步前冷cache，不等于完整模型逐层缓存状态。

## 预先固定的决策规则

- 任一索引、源数据、selected-set完整驻留、fetch字节/scale检查失败：该阶段FAILED/BLOCKED，依赖结论停止；保留原始失败证据。已知问题的修复可以在独立commit下重跑，但不覆盖失败文件。
- long请求若当前配置无法跑到目标长度：记录BLOCKED及OOM/错误，不缩短后继续称256K。
- 初步延迟预算：同一trace的P95新增步管理与搬运时间（C−A的成对步样本）≤无observer基线P50 TPOT的10%。这是工程筛选预算，不是论文阈值；同时报告B−A和C−B，以分离resolve与IO。
- 通过正确性、具备可解释的显存净收益并满足延迟预算：`READ_ONLY_COMPONENT_GO`，仅支持继续开发target-only接入，不能标记部署provisional GO或qualified；它不代表8并发通过。若尚未验证GPU新KV本地驻留、C4闭组、host写回后驱逐与重新fetch、请求释放与重新admit，结论严格限于只读cache组件。
- 正确但超延迟预算：标记“容量用途待评估”，不作为B=1低延迟优化晋级；保留并发容量价值判断，不以统一15%miss率作硬否决。
- 净显存必须扣除热缓存、page table/LRU、尾部/新KV、staging与graph增量；不计入未移动的indexer、GDN或不存在的MTP draft。区分单请求活跃占用、共享总池和8请求预留。

## 接入后的部署资格测试边界

真正声称“支持8条256K并发”需要另一个已实现的target-only QSA host/hot服务路径。本轮若E3通过，可据此冻结接入合同；随后端到端必须测1/4/8条近256K：逻辑总池262144/1048576/2097152、保持相同每请求上限，检查实际准入、cold prefill峰值、输出/attention一致性、稳态TPOT与aggregate吞吐、P95/P99、异常退出及恢复。

该接入资格阶段还必须在同seed/同输入下比较resident与offload的完整768步输出，以及必要的attention/logit参考；覆盖非单位scale、C4边界、prefill、生成KV写回/驱逐/重取和请求回收。E3字节一致只能证明搬运，不能取代这些服务生命周期与数值检查。

在该服务路径尚未存在时这些测试标为DEPENDENCY_NOT_IMPLEMENTED，不能用内存算术、空buffer分配或standalone batch模拟替代。当前8并发结论只是账本支持的验证目标，绝不是已验收能力。

## 产物与状态

所有结果保存本目录；性能数据绑定源码commit。不得计算文件哈希。输入/输出token、两rank trace和原始日志保留。CPU/GPU已运行、失败、未运行逐项记录在最终ANALYSIS中；本协议与任何修订均走git，修订必须在相应新测量之前说明理由。
