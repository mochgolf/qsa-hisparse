# HiSparse V3 integration：完整单请求 eager 验收

最终结论为 **`PASS_V3_SINGLE_REQUEST_EAGER`**。范围严格为一条近256K请求、TP2 eager、真实prefill→host/hot交接→连续decode→释放与同进程再准入，不是8并发验收。

本轮 execution-02 使用候选 source `f04c0d795509c1c52df0147a893a35b634ad20a0`，driver lab `de764d1`；source clean。固定 request-A 261120 输入/768 greedy ignore_eos 输出、seed147342228；maxrun1、max_total/context262144、chunk2048，radix/graphs/overlap/spec/PD 关闭。逐 step tensor capture 关闭，但 handoff/selected byte oracle 与 event 记录保留。

## 已复核证据

- resident/offload各768个输出ID完全一致；本轮resident也与execution-01 resident全部768项一致。每rank767次decode、128次chunk prefill。
- offload每rank12层真实handoff，2340次selected byte/mapping检查；保留C4 tail0/1/2/3及page64边界覆盖。
- offload长请求终止后，在同一candidate进程发2048输入/8输出sentinel：真实raw backing重建、所有层shape与pointer检查通过，然后7次decode、48次selected checks/rank，终止释放成功。
- cancel在真实长RID的handoff_begin发出abort，HTTP200；交接完成后总计2次decode再终止（3个输出，含prefill首输出），不是瞬时抢占。随后同进程sentinel同样恢复raw backing并完成7次decode及释放。
- 五个真实请求（resident long、offload long/sentinel、cancel long/sentinel）的双rank终止均完成postflush，logical_available回到262144；host/hot/workspace归零、Mamba空闲槽恢复40、index-K/Mamba storage未变。健康探测另行保留，不用于替代真实请求证据。

## 每rank内存收益

长请求handoff前raw backing1,611,005,952 B，交接后30,720 B（12层K/V五行ring，槽0保留）。紧邻handoff的CUDA allocated实降1,545,216,000 B，约1.439 GiB/rank；双卡合计约2.878 GiB，但显存不能跨卡合并成单卡容量。

交接后host pinned1,610,612,736 B（1.5 GiB/rank）、GPU hot51,904,512 B、workspace4,229,140 B。index-K201,375,744 B及Mamba2,366,889,984 B保留。handoff logical_available960→960，未提前归还index-K的逻辑lease。

sentinel准入时真实重建raw backing至1,611,005,952 B。正常释放后sentinel前的allocated41429383168→43038310400，增加1,608,927,232 B；取消后的同进程重建增量也为1,608,927,232 B。不是单纯把请求usage归零或依赖进程退出回收。

全arm采样NVML峰值：resident双卡44,284/44,286 MiB，offload45,454/45,452 MiB，cancel45,454/45,452 MiB；后两arm都包含sentinel重新分配，且各有一次nvidia-smi采样超时，保留在原日志与metrics中。它们是采样最大值，不能认作连续时间真实峰值；本轮没有证明全程峰值下降。CUDA allocator释放可复用空间也不要求NVML同步降低。

此收益发生于长prefill完成后的raw-KV交接；prefill仍使用原full raw staging。当前adapter启动guard限定max_running_requests=1，逻辑索引容量也未因raw交接扩大，所以不能由省下1.439GiB直接换算为可用256K并发数。

## 根因修复与验证链

execution-01真实768输出曾从index94分叉。diagnostic-01固定seed后仍复现；第一次decode第一个QSA层已出现K/V差异，且全部坏行都是raw_token%4==0的生成token，整行全零。

根因是adapter给四行ring使用物理槽0，而原有store_cache内核默认将reserved_skip_index0跳过。现保留槽0，用五行物理ring承载四个真实member；handoff尾部、写回、读取及checker全部偏移到槽1..4，继续复用原FP8 writer。source回归5项通过，旧映射negative会失败。

diagnostic-02在相同128输出对照下，双rank127steps×12层Q/indices/metadata/output及四个packed K/V检查点全部exact一致，共39,816字段；本轮再关闭capture，完成原768输出和全部生命周期验收。没有放宽数学容差或缩短最终验收长度。

## 性能与晋级边界

本轮客户端计时（单次验证运行）：

| 指标 | resident | offload |
|---|---:|---:|
| TTFT | 43.157 s | 43.224 s |
| 请求平均TPOT，包含首decode交接间隔 | 94.150 ms | 111.933 ms |
| 首个inter-token gap | 0.124 s | 5.615 s |
| 完整请求e2e | 115.443 s | 129.155 s |

本轮不设延迟门槛。capture关闭不等于所有验证开销关闭：handoff完整字节oracle、周期selected检查和event日志仍在计时内。具体数据见EXECUTION-02-METRICS产物；这些是验证运行的延迟，不能当生产部署吞吐承诺。长请求12层D2H CUDA event累计仅68.268/68.479 ms，计时不包含全部gather/packing/CPU校验与发射间隔，不能用它替代完整交接延迟。长请求handoff双rank约5.421/5.439秒；cancel arm约13.934/13.371秒，存在显著跨arm差异，不将其全部归因于PCIe。

本阶段只覆盖单请求eager的正确性、真实raw backing交接和释放/再准入。尚未获得多请求并发、CUDA graph、radix/prefix sharing、speculation、PD或生产延迟资格。V4 indexed-unpack的B1组件预算证据仍属于独立组件；V3本轮也不能将其升级为8并发服务GO。

## 服务恢复

恢复检查确认 health、启动参数、工作目录、源码和完整环境与预实验快照一致；完整环境只在 driver 内存中比较，未写入公开产物。候选当时尚未合并到 upstream。

独立复审：EXECUTION-02-DATAFLOW-REVIEW.md与EXECUTION-02-LIFECYCLE-REVIEW.md均同意当前窄范围PASS。实现和本轮验收已完成，后续并发/graph/性能扩展需要另立合同。
