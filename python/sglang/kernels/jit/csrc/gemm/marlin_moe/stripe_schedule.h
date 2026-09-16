#pragma once

namespace sglang::device::marlin_moe {

struct whole_k_launch_config {
  static constexpr int thread_k = 64;
  static constexpr int thread_n = 128;
  static constexpr int num_threads = 128;
  static constexpr int blocks_per_sm = 1;
};

// Assign whole output slices to a CTA. Every slice visits K in the same order
// and never participates in a batch-dependent cross-CTA split-K reduction.
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr int whole_k_stripe_iters(int k_tiles, int n_tiles, int parallel, int blocks) {
  return k_tiles * ((n_tiles * parallel + blocks - 1) / blocks);
}

}  // namespace sglang::device::marlin_moe
