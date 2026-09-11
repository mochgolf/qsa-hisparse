# TP2 AutoRound Marlin recovery

## Result

Source `9f8016f5aac9c87d08d01f51761b3874127856a0` restores the intended
Marlin MoE backend on both TP ranks. With the same 38,185-token prompt, B1,
768 output tokens, and no MTP, steady decode increased from 66.768 to 86.374
tok/s, a 29.36% gain. The result is within 0.36% of the previous local Marlin
mean of 86.066 tok/s.

## Root cause and fix

The checkpoint uses symmetric AutoRound groups of 128 values. TP2 splits the
MoE down-projection K dimension to 320 values, which cannot retain whole g128
groups. Latest upstream therefore rejected Marlin and used the slower Triton
MoE path.

Each g128 scale group can be represented exactly as two identical g64 groups.
The fix selects that representation only when g128 fails the existing Marlin
shape check and g64 passes. The loader duplicates scale rows before normal TP
slicing, W2 keeps only its rank-local scale table, and repack releases each
temporary before allocating the next one.

## Validation

- Five real checkpoint experts across gate, up, and down projections were bit
  equal after g128-to-g64 scale expansion.
- The two-rank GPU matrix passed W13 with relative errors 0.00294 and 0.00266,
  and rank-local W2 with 0.00331 and 0.00319. The full-table negative failed at
  1.16 and 1.62 as required.
- Both TP ranks reported Marlin MoE and QSA HiSparse B1; 4,602 CUDA Graph
  decode calls and zero eager fallbacks were recorded.
- KV capacity returned to 2,097,152 tokens, Mamba capacity returned to 40, and
  the service became idle after the requests.
- The fixed clean source passed a guarded 2,048 + 8 token deployment smoke request.

The earlier 8x256K run remains the capacity evidence. It was not repeated for
this MoE-only correction, so its published throughput remains a pre-fix number.
