// Adapted from NEXUS-End2End@23288730907fdb7a8ff7e4683efca35164f3cb8d.
// Arithmetic schedule retained; graph composition belongs to cipheropbench.
#pragma once

#include "Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "cipheropbench/kernels/row_pack.h"

#include <precompiled/torch_includes.h>
namespace cipheropbench::kernels {
using namespace ::nexus;
using namespace std;
using namespace phantom;

class LNEvaluator {
 private:
  std::shared_ptr<CKKSEvaluator> ckks;
  std::shared_ptr<Bootstrapper> bootstrapper;
  void normalize_128x768(std::vector<PhantomCiphertext>& x, std::vector<PhantomCiphertext>& res,
                        const FlatVecArray& weight, const FlatVecArray& bias, bool paired,
                        double variance_bound);

 public:
  LNEvaluator(std::shared_ptr<CKKSEvaluator> ckks, std::shared_ptr<Bootstrapper> bootstrapper) : ckks(ckks), bootstrapper(bootstrapper) {}
  void layer_norm(PhantomCiphertext &x, PhantomCiphertext &res, int len);
  void layer_norm_128x768(std::vector<PhantomCiphertext> &x, std::vector<PhantomCiphertext> &res, const FlatVecArray& weight, const FlatVecArray& bias);
  void layer_norm_paired_128x768(std::vector<PhantomCiphertext>& x, std::vector<PhantomCiphertext>& res,
                               const FlatVecArray& weight, const FlatVecArray& bias,
                               double variance_bound);
};

}
