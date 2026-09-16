# V4 定向 profiling 与优化空间计划

目标是解释 V4 B1 尾延迟的组成、量化可消除成本，并用最少的受控对照验证优化空间。本阶段是性能调查和隔离原型实验，不是生产接入。

## 固定基线与问题

- 沿用已冻结的 integration contract `ad9fdf3`。V1/V2 既有资格保留；V3 不运行。
- V4 baseline 是 attempt-19 的实际脚本 **eb6783ffa3b9c4e5366c16a4c3a9d7b3372851c0**，候选 SGLang **d0664112a8c81dd8019689bc2dd5e5b8fc4f851a**。当前脚本若用于复用，直接 git diff 证明 V4 与实际执行版本一致。禁止计算哈希。
- 双 4090 48GB，TP2，12 层，512 selected C4、2048 hot C4；A/B rank-specific trace，128 单调 steps、request offsets 0/4/.../28，三次完整 reset，双 rank 同步后逐步取慢 rank，再算 nearest-rank P95。
- B1 原 P95 **2.513664 ms**，门槛 **2.427575 ms**，差 **0.086089 ms / 3.55%**。差额是预算缺口，并非一个已定位的操作。B4/B8 原 P95 4.363936/6.790784 ms 仅是 scaling。
- 初步只读重算：B1 三 replay P95 分别 2.529792/2.502208/2.495168 ms，各 replay 最慢五个 step 均为 `step % 4 == 3`。核对真实 position/C4 close 标记再归因；不要将相关性当因果。

## P0：CPU 准备，停服务前完成

1. 阅读基线 V4 的全部调用链、HiSparse resolver/transfer 实现和已有 profiling 工具；复用现成 driver、supervisor、identity oracle、reset、pairing。只增加本轮必要开关/范围，不复制整套框架，不修改历史 attempt 或基线源码。
2. 从原始样本按 replay/rank/step、实际 close/page64 边界分类；保存 P50/P95/P99、慢 rank、逐步配对，选出固定 profiling 窗口。区分冷启动/常态、close/non-close；正式总样本不删异常值或冷启动。
3. 检查可用的 `nsys`、torch profiler 和 NVTX；优先复用已安装工具。CPU self-check 验证新增样本键无重复、配对、分类与统计，验证 instrumentation-off 路径操作/计时范围未改变。保留原有有界异常清理；不重复无关 65536 编码全量测试，除非编码有改动。
4. 提交实验代码、命令和固定实验顺序；文件/源码版本绑定 Git commit。GPU 前再次检查基线身份、其他 GPU owner、测试端口和 NUMA/host 可用内存。

## P1：最小基线与时间线

1. 在一个受控 GPU 窗口运行 instrumentation-off **B1 × 3 reset × 128 step × 双 rank** 基线，保存 event 与 wall time、原始逐步配对数据；记录 GPU clocks/温度/功耗和 CPU/NUMA affinity 等可能影响漂移的实际状态，不锁频、不改系统设置。
2. 用 Nsight Systems 的 CUDA API/GPU/memcpy/NVTX 时间线，仅捕获少量预先选定的 close/non-close 与尾延迟窗口。缓存状态仍从同一个完整 reset 按真实轨迹推进，不能直接跳入被测步。优先同时观察双 rank；工具限制必须写明。
3. 分类：metadata/newest 更新与 tail 输入；producer-gated D2H；resolver 与实际 H2D；selected gather；K/V unpack；GPU idle/CPU launch gaps、event API/等待和动态分配。跟踪 side-stream 与 consumer-stream 的真实依赖及 overlap；不能把 CPU API duration、两 stream 累计 kernel duration、event wait 相加成关键路径。
4. 每个重点 step 保存可回查的时间线位置、总包络、GPU busy 时间并集、idle 时间及依赖解释。特别区分基线中每步从 pinned CPU 注入合成 tail 的 fixture 成本与未来模型真实产生 KV 的工作；不把删掉必要数据生产算成合格优化。
5. 必要时补一个短的分段 CUDA-event 诊断 pass（事件预分配），比较 instrumentation-on/off 开销。profiler/诊断数据仅用于归因，所有预算与收益判定来自 instrumentation-off。不要为 86 us 的缺口使用本身可能超过 86 us 且未经衡量的探针结论。

## P2：证据驱动，最多两个独立优化候选

只针对 P1 已观察的主成本，按改动小/收益大排序。候选方向包括 event/临时 tensor 复用、减少不必要的 Python 发射/小 kernel、gather 与 unpack 的内存遍历或 launch 合并、在保持 D2H→H2D 依赖时减少多余等待。先查已有 helper/已安装 Triton/CUDA 实现；没有时间线证据不编写新 kernel、不重写 resolver。

- 每个候选须写明根因假设、预计减少的工作、正确性风险。保持必要 metadata、tail、writeback、GPU miss resolution/H2D、gather/unpack、依赖与真实 trace；CPU 不提供 miss plan。
- 若候选变更 gather/unpack，加入最小的独立源逐字节检查实际 K/V unpack 输出，至少覆盖被测全部 selected members、request/rank/layer 与 changed-generation；现有 V4 只检查 packed/gathered bytes，不足以自动资格化新的 unpack 实现。若变更 event/transfer，保持 forced non-newest miss、poison negative、variant-1 和原 lifecycle guards。
- 可以用省略操作的对照估计成本上界，但明确标成诊断，不准作为优化通过/预算结果。不得搬出计时范围隐藏仍必须运行的工作。
- B1 固定顺序：baseline 三 reset →短 profiling→每个候选各三 reset→baseline 三 reset（前后 baseline 控制漂移）。每 arm 相同 128 steps、配对与完整 reset；同 step/replay 差值和各 arm 分布都保留。仅比较总 P95 的差不能代替因果定位，不将各阶段 P95 相加。
- 最多两个候选，不进行“跑到通过”为止的搜索。若没有有证据的候选，结束于成本分解和收益上界。只有候选在 B1 显示有用收益且正确性通过，才对最佳候选补 B4/B8 各三 reset，用于观察缩放/回归，不运行模型或声称服务并发资格。

## 安全执行与边界

复用现有 idle/snapshot/finally-restore 协议：基线 running/queue=0 且日志连续 3 秒无 decode/prefill，完整环境仅驻留 controller 内存；测试端口空闲，无其他模型/GPU 工作冲突。结束或失败必须恢复并验证同一个基线。遇到来源变化时停止，避免覆盖用户的新部署。

目标一个 GPU 窗口；整个 profiling GPU subprocess 有界，沿用最多 3900 秒外层上限和有界 peer/进程组清理，另保留服务恢复时间。若工具不可用则使用已有 profiler 的最小替代并降低归因强度；不无限安装/重试。恢复不可被终止 worker 的操作中断。

不运行 V1/V2 全矩阵、V3/full-service，不合并 upstream，不修改基线源码、不增加新依赖或改持久配置。新结果保存在本目录；原 FAIL_BUDGET 不追溯修改。

## 交付与停止条件

交付 `ANALYSIS-RESULTS.md`、manifest、可运行命令/代码提交、原始样本、短 timeline 及解析表、正确性证据、恢复核验。报告应直接回答：

1. 尾延迟主要发生在哪些 steps/rank，close/non-close 相差多少？总时间中传输/内核/CPU 发射与空隙各有什么可证据支持的贡献？
2. 86.089 us 缺口有多大部分可通过已验证操作减少；哪些只是上界或 fixture 特有成本？baseline 漂移是否足以影响预算判断？
3. 最多两个候选各自是否正确、同条件实际节省多少，最佳候选 B1 是否满足原门槛，B4/B8 是否退化？没有通过也如实结束。
4. 下一步最有价值的单个修改是什么，或为何暂不值得继续？结论仍限定组件，不把 V3 依赖和 8×256K 服务验收跳过。

长执行不反复轮询。executor 发送一次完成或失败通知并附恢复状态，不自行触发更多 GPU 运行。
