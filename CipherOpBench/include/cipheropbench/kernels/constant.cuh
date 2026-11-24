// Adapted from NEXUS-End2End@23288730907fdb7a8ff7e4683efca35164f3cb8d.
// Arithmetic schedule retained; graph composition belongs to cipheropbench.
#pragma once
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
#include "cipheropbench/unified_phantom_parameters.cuh"
#endif

namespace cipheropbench::kernels {

#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
constexpr size_t logN = unified_phantom::log_n;
constexpr size_t N = 1ULL << logN, slot_count = unified_phantom::logical_slots;
constexpr size_t L = unified_phantom::data_primes;
constexpr size_t boot_level = unified_phantom::bootstrap_primes;
constexpr size_t total_level = L + boot_level;
constexpr size_t logp = unified_phantom::log_scale, logq = unified_phantom::log_q0;
constexpr size_t log_special_prime = unified_phantom::log_special_prime;
constexpr double scale = unified_phantom::scale;
#else
constexpr size_t logN = 16;
constexpr size_t N = 1ULL << 16;
constexpr size_t slot_count = N / 2;
constexpr size_t L = 16, boot_level = 14, total_level = L + boot_level;
constexpr size_t logp = 40, logq = 51, log_special_prime = 51;
constexpr double scale = 1ULL << logp;
#endif

}  // namespace nexus
