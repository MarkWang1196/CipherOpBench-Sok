// Adapted from NEXUS-End2End@23288730907fdb7a8ff7e4683efca35164f3cb8d.
// Arithmetic schedule retained; graph composition belongs to cipheropbench.
#pragma once

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"

namespace cipheropbench::kernels {
using namespace ::nexus;
using namespace std;
using namespace phantom;

class ArgmaxEvaluator {
 private:
  std::shared_ptr<CKKSEvaluator> ckks;
  std::shared_ptr<Bootstrapper> bootstrapper;

 public:
  ArgmaxEvaluator(std::shared_ptr<CKKSEvaluator> ckks, std::shared_ptr<Bootstrapper> bootstrapper, int main_mod_count) {
    this->ckks = ckks;
    this->bootstrapper = bootstrapper;
  }

  void argmax(PhantomCiphertext &x, PhantomCiphertext &res, int len);
};
}  // namespace nexus
