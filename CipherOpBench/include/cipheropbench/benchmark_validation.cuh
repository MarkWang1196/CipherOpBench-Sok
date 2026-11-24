#pragma once
namespace cipheropbench::detail {
inline bool can_time_method(bool finite,bool ckks_matches,bool exact_matches,bool has_noiseless_reference) noexcept {
 return finite&&ckks_matches&&(exact_matches||has_noiseless_reference);
}
}
