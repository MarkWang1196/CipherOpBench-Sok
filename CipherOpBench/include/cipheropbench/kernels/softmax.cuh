// Adapted from NEXUS-End2End@23288730907fdb7a8ff7e4683efca35164f3cb8d.
// Arithmetic schedule retained; graph composition belongs to cipheropbench.
#pragma once

#include "Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "cipheropbench/kernels/row_pack.h"
#include "phantom.h"

namespace cipheropbench::kernels {
using namespace ::nexus;
using namespace std;
using namespace phantom;

class SoftmaxEvaluator {
 private:
  std::shared_ptr<CKKSEvaluator> ckks;
  std::shared_ptr<Bootstrapper> bootstrapper;

 public:
  SoftmaxEvaluator(std::shared_ptr<CKKSEvaluator> ckks, std::shared_ptr<Bootstrapper> bootstrapper) : ckks(ckks), bootstrapper(bootstrapper) {}

  void softmax(PhantomCiphertext &x, PhantomCiphertext &res, int len);
  void softmax_128x128(PhantomCiphertext &x, PhantomCiphertext &res, FlatVec attention_mask);
};
}  // namespace nexus
