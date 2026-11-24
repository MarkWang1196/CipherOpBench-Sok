#pragma once
#include "ckks_evaluator.cuh"
#include "Bootstrapper.cuh"
#include "cipheropbench/kernels/nexus_utility.cuh"
#include "cipheropbench/calibration.hpp"
#include <chrono>
#include <functional>

namespace cipheropbench {
namespace k = kernels;
using Cipher = PhantomCiphertext;
using CipherList = std::vector<Cipher>;
struct Event {
  std::string name;
  double milliseconds;
  size_t input_chain = 0, output_chain = 0;
};
class Runtime {
public:
  std::shared_ptr<nexus::CKKSEvaluator> ckks;
  std::shared_ptr<::Bootstrapper> bootstrapper;
  std::vector<Event> events;
  bool tracing = false;
  std::function<void(const Cipher&,const Cipher&)> refresh_observer;
  calibration::Profile calibration;
  std::function<void(size_t,size_t)> layer_completed;
  explicit Runtime(bool with_bootstrap);
  Runtime(bool with_bootstrap, const std::vector<int>& rotation_steps);
  static const std::string& input_encryption_mode();
  void synchronize() const;
  Cipher encrypt(k::FlatVec values, size_t chain = k::boot_level + 1) const;
  CipherList encrypt_hidden(torch::Tensor input, size_t chain = k::boot_level + 1) const;
  torch::Tensor decrypt_hidden(const CipherList& input) const;
  torch::Tensor decrypt_slots(const Cipher& input) const;
  void refresh(Cipher& input);
  void refresh(CipherList& input);
  void record(const std::string& name, const std::function<void()>& fn,
              size_t input_chain = 0, const Cipher* output = nullptr);
};
}
