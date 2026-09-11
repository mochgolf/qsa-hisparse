# P0 execution-01：正确性及轻量端到端基线

候选源码 `9f494dca71771acedad2ef437272a2f281fbfed8`，执行 driver lab `45b60da`。复核结论为 `PASS_P0_CORRECTNESS_AND_E2E_ONLY`；组件归因未完成，不是完整P0性能归因或多并发资格。

## 测量结果

同源TP2/eager/FA2、固定seed、261120输入及768 greedy输出，light三组成对顺序R/O、O/R、R/O。以下是三次运行的对应指标中位数；P95栏是各次请求P95的中位数，不是混池P95。

| 指标 | resident light | offload light |
|---|---:|---:|
| TTFT | 43.127秒 | 43.126秒 |
| steady平均输出间隔，排除首次交接间隔 | 93.820毫秒 | 97.036毫秒 |
| steady P95 | 94.488毫秒 | 98.277毫秒 |
| 首次输出间隔 | 0.130秒 | 1.376秒 |
| 请求e2e | 115.192秒 | 118.884秒 |

各组offload−resident steady开销为2.660、3.894、3.216毫秒/token，中位数3.216毫秒，约为resident中位数的3.43%。三组方向一致，均超过0.641毫秒的工程筛选阈值。原reducer的 `UNRESOLVED` 表示“未检出正向性能收益”，不是“无法看出offload更慢”；它是单向收益筛选，未做统计显著性或生产SLO声明。六条请求全部767个token间隔可分辨，没有SSE合并，steady每条766个间隔。

prefill约6063 token/s，TTFT基本相同。light offload真实handoff每次较慢rank分别1260.702、1233.370、1240.180毫秒；D2H事件累计每rank约66.6–66.8毫秒。两者差额不能全部归为CPU或PCIe空闲：当前尚缺gather/pack、分配、同步、发射等分项证据。

本轮strict offload较慢rank的handoff为2568.030毫秒，完整请求TPOT为108.380毫秒，首次间隔2.732秒；strict resident完整TPOT94.622毫秒。这与上一轮严格交接约5.44秒存在明显跨运行漂移，不能把旧5.44秒到当前light1.24秒全部归功于移除oracle，也不能混用包含首次交接的TPOT和steady间隔。

## 正确性、容量与恢复

strict沿用原完整resident/offload/cancel合同和独立K/V oracle；light另用独立生命周期reducer。八个完整长请求均与execution-02 resident的768输出ID完全相同。light关闭高频oracle而保留身份、结构及事件依赖；offload后同进程sentinel、strict取消后释放/再准入均通过。严格长请求每rank2340次selected检查，sentinel每rank48次；tail/page/mapping覆盖成立。全部真实请求最终logical恢复262144、Mamba空闲恢复40，index/Mamba backing不变；offload终止host/hot/workspace归零。独立证据见 `P0-RESULT-LIFECYCLE-REVIEW.md` 和 `P0-RESULT-MEASUREMENT-REVIEW.md`。

保留原单请求显存收益：strict长请求handoff raw backing归为30720 B/rank，allocated实降1545216000 B/rank；host为1.5 GiB/rank，hot为49.5 MiB/rank，workspace4229140 B/rank。light采样NVML峰值resident每arm44286 MiB、offload每arm45454 MiB，后者包含sentinel重建；不是连续时间绝对峰值，也没有证明全程峰值降低。

实验后服务恢复检查通过，driver 与独立复核均确认 health、启动参数、工作目录、源码和选定环境与预实验快照一致；完整环境只在 driver 内存中比较，未写入公开产物。

## 下一步

继续已授权P0的组件归因：复用现有profiler/NVTX机制，在真实长请求decode窗口标记indexer、resolve/refetch、gather/unpack/compact、FA2和handoff阶段。观测补丁须通过去除注解后与9f基线AST等价检查及原CPU回归；profile结果用于定位，独立于轻量性能基线。完成归因再按P1优先级优化，不提前做多请求prefill，不据本结果授予8并发GO。
