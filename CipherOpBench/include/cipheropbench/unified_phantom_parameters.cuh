#pragma once
#include <array>
#include <cstdint>
#include <vector>
namespace cipheropbench::unified_phantom {
// Functional candidate only; security estimation and RNG audit remain pending.
// Every method in a unified build must compile against this same profile.
inline constexpr std::array<uint64_t,35> coefficient_primes{2251799710138369ULL,281474926903297ULL,281474931621889ULL,281474933456897ULL,281474938175489ULL,281474941059073ULL,281474945253377ULL,281474948136961ULL,281474948923393ULL,281474949185537ULL,281474955476993ULL,281474962554881ULL,281474975662081ULL,2251799714856961ULL,2251799716691969ULL,2251799726129153ULL,2251799726653441ULL,2251799741857793ULL,2251799756013569ULL,2251799756800001ULL,2251799757586433ULL,2251799763091457ULL,2251799765975041ULL,2251799772266497ULL,2251799774887937ULL,2251799776198657ULL,2251799777771521ULL,2251799780917249ULL,2251799787995137ULL,2251799789568001ULL,2251799791403009ULL,2251799797432321ULL,2251799799267329ULL,2251799806345217ULL,2251799807131649ULL};
constexpr int log_n=17, log_scale=48, data_primes=12, bootstrap_primes=21;
constexpr int log_q0=51, log_bootstrap_prime=51, log_special_prime=51;
constexpr int secret_hamming_weight=192;
constexpr int logical_log_slots=15,logical_slots=1<<logical_log_slots,physical_slots=1<<(log_n-1);
inline std::vector<int> rotation_steps(){std::vector<int> steps{0};for(int i=0;i<log_n-1;++i)steps.push_back(1<<i);steps.insert(steps.end(),{-1,-32,-1024});return steps;}
constexpr int bootstrap_log_width=4, bootstrap_inverse_degree=15;
constexpr int working_moduli=data_primes+1, input_chain_index=bootstrap_primes+1;
constexpr int bootstrap_modular_degree=79;
constexpr int precision_refresh_gain=256, physical_bootstraps_per_refresh=2;
constexpr double refresh_amplitude=2.;
constexpr double scale=281474976710656.;
}
