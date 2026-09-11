# QSA → HiSparse：双 RTX 4090 48GB 部署收益评估

本报告先整理五档 CPU 回放；该阶段尚无 QSA-HiSparse GPU 原型或端到端 A/B，不能标记部署 qualified。

读取了前期设计讨论并逐项核查，未将讨论中的建议当作已完成实验。此次只读检查实验服务并做 CPU 数据分析；未启动生成、停止或重启服务，也未修改模型或基线源码。

**部署判断：单请求低延迟用途优先级低；有实际长上下文并发容量需求时，值得做受限原型。现有证据支持核算显存收益和选择局部性，不支持承诺正吞吐收益。**

## 1. 实验设备与版本

运行时快照只用于核对下列实验配置，不公开本机进程、端口或路径。

- GPU：2 × RTX 4090，每卡驱动报告 49140 MiB；PCIe Gen4 ×16；GPU 间拓扑 SYS，分别邻近 NUMA 3、2。
- CPU：AMD EPYC 9654，96 个在线 CPU；内存约 566 GiB。Host 容量足够容纳本轮预计的少量 GiB KV，但不等于随机 pinned-host 读取延迟已经合格。
- P2P 本地缓存探测记录双向为 true；本次未重跑 P2P 测试。引用会话最早的 P2P=false 不宜继续作为当前事实。P2P 与 GPU 读取 CPU pinned memory 是不同路径。
- 基线源码：`6f25e04479152cfe68e76a16b5c00236a6f360f8`，采集时源码树 clean。
- 模型：W4A16 RTN AutoRound INT8PLE；TP2，FP8 E4M3 KV，FP32 GDN state；**speculative_algorithm=null**。
- context/max_total_tokens 均 262144，max_running_requests=8，max_mamba_cache_size=40；确定性推理，decode CUDA Graph full，prefill graph disabled，page size 64，radix cache 实际关闭。

运行时启动账本（每卡，数值按实现的二进制 GiB 解释）：weight=36.869，kvcache=1.688，decode graph=0.230，startup_available=3.471。它不是完整的即时显存分配审计，各项不可无条件相加闭合。GDN conv=0.04、SSM=2.16，raw K/V 各 0.75。

## 2. 可释放的是哪一部分

QSA 的 C4 指 indexer 每四个原 token 做一次选择压缩；attention 仍读取这四个原 token 的 K/V。它不是 DeepSeek V4 那种可以直接复用布局的单条 compressed MLA KV。

本地实现：`python/sglang/srt/mem_cache/qsa_kv_pool.py` 的 QSATokenToKVPool，及 QSA indexer 的 block expansion 路径。12 个 QSA 层；TP2 后每卡一个 KV head，head_dim=256。

每卡每层每原 token：K+V = 2 × 256 × 1 = 512 B；每 C4 block=2048 B。12 层合计 **6 KiB / 原 token / GPU**。

BF16 indexer 为每四 token 一个 128 维 key，且本配置在 TP ranks 复制：12 × 128 × 2 / 4 = **768 B / 原 token / GPU**。262144 token 时 raw KV=1536 MiB，indexer=192 MiB，合计1728 MiB=1.6875 GiB，与实时账本匹配。还存在少量 padding、pending ring 等。

| 缓存 C，C4 blocks/层/请求 | 原 token slots | 热 KV / 请求 / 卡 | 单个满 256K 请求毛节省 | 为8请求预留热KV时，相对当前共享256K池的毛节省 |
|---:|---:|---:|---:|---:|
| 512（1×） | 2048 | 12 MiB | 1524 MiB | 1440 MiB |
| 1024（2×） | 4096 | 24 MiB | 1512 MiB | 1344 MiB |
| 1536（3×） | 6144 | 36 MiB | 1500 MiB | 1248 MiB |
| 2048（4×） | 8192 | 48 MiB | 1488 MiB | 1152 MiB |
| 4096（8×） | 16384 | 96 MiB | 1440 MiB | 768 MiB |

以上均是布局推导的**毛节省**，不是已释放显存。净节省须扣除新增 page table、LRU、末尾未完成 C4 block、新 KV 本地驻留与写回缓冲、prefill 暂存区及其他元数据。若旧 GPU KV pool 仍完整保留，只增设 host pool 与 hot cache，则不会产生上述节省。

关键分母：当前 262144 是8个请求共享的总池。不能把8×1.5GiB称为当前部署节省。反过来，单条38185-token trace的活跃raw KV只有223.74MiB/卡，4×热缓存相对这条活跃历史毛省175.74MiB；回收整个预分配池才可能接近上表单请求的1488MiB。

2×–4×热缓存下，单请求回收整个现有 raw KV pool 的上限约1.45–1.48GiB/卡，占每卡总显存约3.1%；预留8个热缓存时约1.125–1.3125GiB/卡，双卡约2.25–2.625GiB，尚未扣元数据。

GDN不参与offload：单个live state理论约55MiB/卡，而当前40容量的池加padding已分配约2.20GiB/卡。用当前已分配池口径，4×缓存后，GDN占「GDN池+indexer+热KV」约90%（一份热缓存）或80%（八份热缓存）。这不是整个模型显存占比，也不能套会话中MTP的0.3–0.5GiB/request估计。当前不存在可额外节省的BF16 draft KV。

## 3. 速度与容量要分开判断

原始 r1 采集的有效数据是第四窗口、38185 输入 token、768 输出、767 decode 步，source commit 与基线一致。rank0/rank1 集合相同，第五窗口只是重复性核对，不能作为独立任务样本。

1×缓存首步初始化后，miss = 1,579,601 / 4,706,304 = 33.56351396%。因此纯selection-demand流量均值为 **4.02762MiB / 输出token / GPU**（12层求和）；两卡各自搬本rank的KV。新生成KV在GPU产生，真正host→device历史miss还需单列，不能把全部selection miss都当作最终H2D实测。

本次新增CPU回放：[replay_lru.py](replay_lru.py)、[replay_summary.json](replay_summary.json)。每层独立cache，首个selection预装，按完整selected set判断hit/miss，保护本步选中项，旧hit优先于新miss成为MRU；组内按block ID升序确定顺序。五档使用完全相同轨迹，**不是五组独立任务样本**。

| 热缓存/层/请求 | 回放miss比例 | 均值 MiB/token/GPU | P95 MiB/token/GPU | P99 MiB/token/GPU |
|---:|---:|---:|---:|---:|
| 1×：512 C4 | 33.564% | 4.028 | 5.908 | 6.679 |
| 2×：1024 C4 | 16.564% | 1.988 | 3.474 | 4.109 |
| 3×：1536 C4 | 10.307% | 1.237 | 2.428 | 3.037 |
| 4×：2048 C4 | 7.107% | 0.853 | 1.735 | 2.340 |
| 8×：4096 C4 | 2.414% | 0.290 | 0.714 | 1.514 |

分位数先对同token的12层miss求和，再求分位数。全部以首步之后766步为统计窗口，包含大缓存逐渐填充的阶段；不是“前128步剔除后”的稳态值。以上字节仅是2KiB/C4的选择需求换算，不是PCIe计数器数据。初始装入、生成新KV本地驻留和host写回没有纳入这个简化需求cache模型，不能把它称为完整HiSparse生命周期回放或逐字节传输仿真。排序策略明确，但没有证明与某个CUDA实现的所有内部tie-break完全一致。原始top-k集合合法且可重复，也不替代真实logits的top-k correctness哨兵。

这比会话里的“假设2×能降至13%–15%”更有依据：本地结果是**2×16.56%、4×7.11%**。若做单档搬运筛选，4×是合理首测候选：比2×多24MiB/请求/卡，均值需求流量少约57%；仍需近256K轨迹与真实resolve延迟决定最终容量，不能把miss曲线当作性能放行。

**64-token整页搬运不适合直接套在上述预算上。** 每页含16个C4 block，同等显存预算只有32/64/96/128/256页。实际每层每步selected set需要87–278页（均值168.18；首步后窗口）。因此五档容量都存在装不下本步完整selected page set的步骤，4×预算128页尤其不足。这个需求下界已经能否定“同预算整页cache可覆盖整条轨迹”，无需伪造一次可行整页LRU。它不否定更大的整页cache，也不要求修改现有page=64的逻辑分配器；可在其内部按C4子块搬运。JSON中的miss_pages字段只统计C4策略miss落在哪些页，**不是整页LRU miss率**。

这点流量的持续带宽通常不是最严峻的问题，但12层串行的resolve、host load和同步会直接进入token关键路径。例：仅为敏感性算术，4.03MiB若能达到20GB/s有效载荷带宽，payload时间约0.21ms；这不包括每层固定开销、随机访存、NUMA、双卡竞争、尾延迟，也不是本机测量值。两卡时延不能相加，TP步时延取决于较慢rank及同步。

QSA每层独立算selection，不能直接获得HiSparse在共享indexer模型上的exact跨层prefetch收益；也不能用MTP IndexShare的接受率推断target跨步hit rate。[HiSparse论文](https://arxiv.org/html/2608.07009v1)中的4.7×峰值吞吐来自不同模型和H200/B200/GH200容量受限负载；[官方介绍](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)明确指出低并发有额外IO代价。这里没有理由把offload本身当成单请求attention算力加速。

同设备已有更直接的低延迟证据：QSA service A/B 在同一 38K 性能任务上从 82.481 到 104.931 tok/s（+27.22%，12.124→9.530 ms/token），通过改进 SM89 attention 并行度取得，source `bd83f022...`→`3f1a20c8...`。B>1 不在验证范围；这组速度不能当作 HiSparse A/B 基线。

当前日志的last_gen_throughput同样是其他实验请求，不能当无observer稳态基线。本次没有测得HiSparse后的tok/s、TPOT或TTFT；任何正加速承诺都超出证据。

## 4. 真正可能获益的部署范围

| 需求 | 判断 | 原因 |
|---|---|---|
| 单个约38K编程请求更快 | 低优先级 | 活跃raw KV小；多了host miss关键路径；已有QSA算子优化证据更直接 |
| 单个接近256K请求能稳定装入 | 有限容量价值 | 能提供约1.45GiB/卡毛余量，但prefill/graph峰值不必同步下降 |
| 多个长上下文同时decode | 值得受限原型 | 可以把raw KV residency从总历史量改为每请求固定working set |
| 直接取得2×–5×吞吐 | 无证据 | 需额外并发有效利用GPU、且不被prefill/权重/PCIe限制 |
| 直接支持1M上下文 | 不成立 | 模型与服务当前context上限262144，offload不扩展模型有效上下文 |

例如未来8个各256K历史的容量算术：full raw KV=12GiB/卡；4×热KV=0.375GiB/卡；indexer仍需1.5GiB/卡。raw KV毛减11.625GiB/卡具有吸引力，但这是新2M逻辑token总池的比较，不是当前省11.625GiB，也不是8并发已可部署。还需逻辑地址空间、scheduler admission、host池、GDN槽位、prefill暂存/工作区和B>1 backend共同通过。

max_total_tokens 是显式 262144 上限；仅节省显存而不调整逻辑 token 分配与调度，长请求准入能力仍不会自动增长。可用显存 3.47 GiB 是启动时余量，不是安全增加池容量的凭证；同一设备上已有确定性服务启动/热身 OOM 记录。

## 5. 移植边界与最小下一步

基线 `arg_groups/hisparse_hook.py:98` 明确只接受 DSA 或 DeepSeekV4，QSA 不能仅添加 `--enable-hisparse`。现有 QSA compressed-slot 地址由 full-slot 整除 4 推导；缩小 GPU pool 时必须把逻辑历史地址与 hot physical slot 分开，保留 indexer 的完整历史寻址。K/V 是独立 GQA 布局，FP8 scale 与现有 gather 语义要保持；原有 DSA/DSv4 pool 不能按名称直接替换。

当前[官方指南](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hisparse_guide.mdx)列出DSA/DSv4并强调PD decode部署，而论文/早期博客还展示colocated实验；这里按固定源码版本的实际能力判断，不能把论文系统能力等同于该配置即开即用。

最小有决策价值的后续GPU工作是：两类真实近256K trace（含top-k correctness哨兵），然后只对一档候选缓存做双卡、真实NUMA、真实FP8布局、CUDA Graph、12层串行的resident/resolve-only/resolve+host-fetch比较。CPU预先算好miss只能证明fetch-only，不能证明GPU resolve开销。

工程筛选可设P95新增步延迟≤关闭observer的同配置基线P50 TPOT的10%；这是待采用的预算，不是论文定律或已通过门槛。微基准通过后才做target-only端到端A/B，同时检查实际净释放显存、cold prefill、并发准入和输出一致性。2×/4×缓存大小应由本地回放和双卡延迟共同决定，不以通用15%miss阈值放行。

对于当前任务，CPU回放与账本已能完成收益评估；大规模GPU实验、MTP、FP8 indexer及生产接入不是这份报告已完成的内容。

## 6. 复核与复现

独立源码审查确认raw KV/indexer账本、C4展开含义、GDN池口径、QSA不在现有HiSparse接入范围。主侧审查回放更新顺序，并运行synthetic自检验证「hit优先miss」和「本步selected set不被驱逐」；五档回放重新运行成功，1×精确恒等式通过。

实际CPU输入使用同窗口独立rank0文件；主侧另逐行验证它与原始tp01中rank0的9204行完全一致，见[provenance_check.json](provenance_check.json)。未计算文件哈希；rank1和第五窗口不计作额外样本。

复现命令（只用CPU，输出写临时文件）：

```bash
python3 experiments/assessment/replay_lru.py \
  /path/to/qsa-rank0-trace.jsonl \
  --output /tmp/qsa-hisparse-replay-summary.json
```
