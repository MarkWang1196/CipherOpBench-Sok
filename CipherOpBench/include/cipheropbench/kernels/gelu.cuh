// Adapted from NEXUS-End2End@23288730907fdb7a8ff7e4683efca35164f3cb8d.
// Arithmetic schedule retained; graph composition belongs to cipheropbench.
#pragma once

#include "Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "phantom.h"

namespace cipheropbench::kernels {
using namespace ::nexus;
using namespace std;
using namespace phantom;

class GELUEvaluator {
 private:
  int d_g = 2;
  int d_f = 2;

  std::shared_ptr<CKKSEvaluator> ckks;
  std::shared_ptr<Bootstrapper> bootstrapper;

 public:
  GELUEvaluator(std::shared_ptr<CKKSEvaluator> ckks, std::shared_ptr<Bootstrapper> bootstrapper) : ckks(ckks), bootstrapper(bootstrapper) {}

  void gelu(PhantomCiphertext &x, PhantomCiphertext &res);
};
}  // namespace nexus
