# V4 定向 profiling 与优化空间结果

> fused-unpack 的隔离组件正确性与 B1 预算结果可接受。收益是本窗口观测值；event-reuse 的约 35 us 降幅小于基线漂移，不能称稳定收益。fused-unpack 降低所有步骤共有的 unpack 成本，close 额外延迟仍约 0.74 ms；B8 主瓶颈尚未独立定位。

本轮完成冻结计划 `26abab1` 的唯一受控 GPU 窗口。实验脚本提交为 `0e8c036`，attempt-19 实际 V4 脚本为 `eb6783f`，候选 SGLang 为 `d066411`。CPU preflight 逐函数确认当前 `v4()` 与 attempt-19 一致；未修改历史 attempt 或候选 SGLang。

## 结论

V4 B1 尾延迟由 C4 close 步主导，但不是一次 86.089 us 的单 kernel 问题。历史 384 个配对样本中，96 个 close 样本 P50/P95 为 2.493568/2.538944 ms，288 个 non-close 为 1.742368/1.826752 ms，最慢 20 个样本全部是 close。新窗口的 close miss 数量反而略低于 non-close，因此更大的自然 H2D miss plan 不能解释 close 尾部。

Nsight Systems 显示，profiled close 步与 non-close 步的 GPU busy 并集分别只有 443.510/460.573 us，而 GPU 包络为 3828.260/2827.706 us，包络内 idle 为 3384.750/2367.132 us。close 增加约 1017.617 us idle，GPU busy 没有增加。主要差别在 eager CPU 发射、每层 D2H 的 event/stream 编排和由此形成的 launch gap；实际 24 KiB D2H 并不是一个 0.7–1.0 ms GPU copy。

两个有时间线证据的最小候选均保持 correctness。event 复用只获得 0.034784 ms 的保守 P95 收益，覆盖历史缺口的 40.4%，B1 P95 仍为 2.513888 ms，未过 2.427575 ms 门槛。将每层 8 次 K/V unpack `copy_` 替换为一次预索引 `index_select`，B1 P95 降至 **1.783616 ms**，保守减少 **0.765056 ms**，以 **0.643959 ms** 裕量满足原门槛。该候选没有省略 gather/unpack 或把工作移出计时范围。

这是组件隔离候选的 `PASS_B1_BUDGET`，不追溯修改 attempt-19 的 `FAIL_BUDGET`，也不解决 V3 allocator/service dependency 或 8×256K 服务验收。

## 基线分层与漂移

| arm | 配对 P50 | 配对 P95 | 配对 P99 | close P95 | non-close P95 | 门槛裕量 |
|---|---:|---:|---:|---:|---:|---:|
| attempt-19 | 1.788672 | 2.513664 | 2.539904 | 2.538944 | 1.826752 | -0.086089 ms |
| baseline-before | 1.868096 | 2.548672 | 2.583648 | 2.576608 | 1.915392 | -0.121097 ms |
| baseline-after | 1.928672 | 2.595808 | 2.630784 | 2.616736 | 1.964960 | -0.168233 ms |
| event-reuse | 1.895136 | 2.513888 | 2.549472 | 2.532896 | 1.933504 | -0.086313 ms |
| fused-unpack | 1.125472 | **1.783616** | 1.821376 | 1.806176 | 1.154080 | **+0.643959 ms** |

新前后 baseline P95 相差 0.047136 ms，是历史 0.086089 ms 缺口的 54.75%；两者相对 attempt-19 分别慢 0.035008/0.082144 ms。这个漂移足以影响接近门槛的 event-reuse 判断，但远小于 fused-unpack 的 0.765056 ms 保守收益。正式样本保留了 baseline-before replay 0 step 0 的 16.085184 ms 冷启动异常；删除它不影响本轮 P95，报告和原始数据均未删除。

baseline-before 的 96 个 close 样本平均 miss count 为 1038.94，288 个 non-close 为 1142.71；P95 分别为 2071/2098。page64 是 close 的子集，六个样本 P95 2.543936 ms，没有观察到比普通 close 更高的独立阶跃。前基线慢 rank 分布为 rank0 311/rank1 73，后基线变为 rank0 102/rank1 282，说明 rank 侧漂移真实存在，不能用单 rank 或 `max(rank P95)` 代替逐步配对。

## 时间线分解

profile 从同一完整 reset 沿真实轨迹推进至 step 72，再捕获 step 72–79；其中 step 75/79（position 261195/261199）为 close，其余六步为 non-close。双 rank 共 16 个 step range、960 个 layer-stage range。下表是每个完整 step 的均值；GPU busy 是跨 stream 区间并集，不重复相加重叠工作。

| 类别 | CPU NVTX 包络 | CUDA API | GPU busy 并集 | GPU 包络 | 包络内 idle | kernels | memcpys |
|---|---:|---:|---:|---:|---:|---:|---:|
| close | 3906.639 us | 976.430 us | 443.510 us | 3828.260 us | 3384.750 us | 132 | 60 |
| non-close | 2927.757 us | 765.372 us | 460.573 us | 2827.706 us | 2367.132 us | 132 | 48 |
| close − non-close | +978.882 us | +211.058 us | -17.063 us | +1000.555 us | +1017.617 us | 0 | +12 |

每个 non-close layer 的 metadata/tail 区域恰有 4104 bytes H2D：2048-byte top-k、4-byte seq、2048-byte合成 tail record 和 4-byte newest mapping；close 再增加一个 2048-byte D2H，所以是 6152 bytes。合成 tail record 的 H2D 是 fixture 每步注入未来模型产生 KV 的成本，不能当作生产优化删除；本候选保留了它。close 每层 metadata/tail CPU range 从 52.456 us 升至 116.272 us，显著大于对应 GPU busy 从 1.618 us 到 2.844 us 的增加。

close 中 CUDA event create/record/wait API 每步平均 112.566 us，non-close 为 13.985 us，差约 98.581 us；这支持 event-reuse 候选，但 instrumentation-off 实测表明本窗口观测到的保守 P95 差值只有 34.784 us，小于前后基线漂移，尚不能称稳定可回收成本。未把 API duration 当成 GPU kernel 成本。

baseline unpack 每层固定发出 8 个 `elementwise_kernel`，每个 rank-step 共 96 个；其每层 CPU range/CUDA API/GPU busy 均值约为 106.968/27.034/14.755 us。resolver 每层一个 `load_cache_to_device_buffer_kernel`，gather 每层一个 dtype cast 加一个 `vectorized_gather_kernel`。这些 stage 数字只用于工作归属；跨 stream 和 CPU/GPU 有重叠，不能把 12 层或阶段 P95 相加成关键路径。

Nsight + NVTX 明显改变了绝对时间：相对 instrumentation-off 前基线同 step 中位数，profiled close/non-close event time 平均分别增加 1.327/1.061 ms，均远大于 86.089 us。因此所有预算和候选收益只使用 instrumentation-off 结果；时间线只支持结构归因。

## 候选结果与正确性

`event-reuse` 在计时外为每个 request/layer 预分配 producer/done event，计时内仍记录 producer、side-stream wait、D2H、done 和 consumer wait。其总 P95 相对较快的 bracketing baseline 仅减少 0.034784 ms；同键 close 中位收益相对前后 baseline 为 0.033312/0.082608 ms。它正确但不足以单独通过门槛，也说明 close 的全部差值不能归于 event allocation。

`fused-unpack` 预计算 2048 个 K 行和 2048 个 V 行的 chunk 索引，用一次现有 PyTorch `index_select` 生成连续的 K/V 输出视图，将每层 8 个 launch 降为 1 个；gather、实际 unpack 字节量、D2H→H2D 依赖和 resolver 均保留。相对前/后 baseline 的同键 close 中位收益为 0.770240/0.815840 ms，non-close 中位收益为 0.761536/0.806432 ms，与“固定减少 unpack launch”一致。

V4 历史只检查 packed/gathered bytes，本轮为实际 unpack 补了独立 immutable identity oracle。B1 fused-unpack 双 rank 共完成 72 个 forced variant-1 writeback/displace/refetch/poison lifecycle 检查、4896 个实际 K/V unpack byte checks，逐字节覆盖 2,566,914,048 bytes，包括全部 selected members、request/rank/layer、验证 step 和 changed generation。B4/B8 分别完成 288/576 个 forced lifecycle 检查、19,584/39,168 个 unpack checks，覆盖 10,267,656,192/20,535,312,384 bytes。所有 rank 状态均为 `PASS_COMPONENT_CORRECTNESS`。

## B4/B8 scaling

只有最佳 fused-unpack 候选运行了 B4/B8，各为 3 reset × 128 steps × 双 rank。

| B | fused-unpack P50 | fused-unpack P95 | fused-unpack P99 | attempt-19 P95 | 历史差值 |
|---:|---:|---:|---:|---:|---:|
| 4 | 1.894080 | 3.693920 | 3.891936 | 4.363936 | -0.670016 ms (-15.35%) |
| 8 | 3.198144 | 6.555008 | 6.776960 | 6.790784 | -0.235776 ms (-3.47%) |

B4/B8 没有同窗口 baseline；历史比较只表明未观察到 P95 回归，不能把差值全部归因于候选，也不能当作服务并发结论。B8 的收益比例明显收窄，提示 miss/H2D 或其他 batch scaling 成本值得关注；没有该 B 点的时间线，尚不能确定主瓶颈。

## 下一步与边界

下一步最有价值的单个修改，是把 `fused-unpack` 的一次预索引 copy 接到实际 V4 组件路径，并复用固定索引/output workspace；本轮隔离实现已经给出充足正确性和 0.644 ms 预算裕量。暂不组合 event-reuse：它单独只回收约 35 us，而 fused-unpack 已远离门槛；实际接入后若 profile 仍显示 close tail 接近预算，再加 event 复用即可。

本轮没有运行 V1/V2 全矩阵、V3、模型或 full-service，没有改门槛、锁频、服务配置或候选源码。运行前 idle gate 确认 running=0/queue=0、连续 3 秒无 decode/prefill，测试端口与 GPU owner 均为空。GPU 阶段共 185.849 s。恢复后的 health、启动参数、工作目录、源码和环境均与预实验快照匹配。初次 controller 曾因比较 `/v1/models.created` 动态启动时间戳误报 restore mismatch；原误报保留在私有 manifest，稳定身份复核已纠正为 complete。

证据入口：[execution manifest](execution-01/manifest.json)、[CPU preflight](cpu-preflight.json)、[analysis summary](execution-01/analysis-summary.json)、[raw rank samples](execution-01/raw-samples.jsonl)、[paired samples](execution-01/paired-samples.jsonl)、[timeline analysis](execution-01/timeline/timeline-analysis.json)、[timeline ranges](execution-01/timeline/timeline-events.jsonl)、[Nsight report](execution-01/timeline/b1.nsys-rep)、[candidate selection](execution-01/candidate-selection.json)、[restore verification](execution-01/restore-verification.json)。
