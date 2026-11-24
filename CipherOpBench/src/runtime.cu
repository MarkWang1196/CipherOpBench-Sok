#include "cipheropbench/runtime.cuh"
#include "cipheropbench/legacy_precision_refresh.cuh"
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
#include "cipheropbench/unified_refresh.cuh"
#endif
#include <stdexcept>
#include "cipheropbench/gpu_reservation.cuh"

namespace cipheropbench {
const std::string& Runtime::input_encryption_mode(){
 static const std::string mode=[]{
  const char* value=std::getenv("CIPHEROPBENCH_INPUT_ENCRYPTION");std::string selected=value?value:"public_key";
  if(selected!="public_key"&&selected!="client_symmetric")throw std::invalid_argument("CIPHEROPBENCH_INPUT_ENCRYPTION must be public_key or client_symmetric");
  return selected;
 }();return mode;
}
Runtime::Runtime(bool with_bootstrap):Runtime(with_bootstrap,{}) {}
Runtime::Runtime(bool with_bootstrap,const std::vector<int>& rotation_steps) {
  (void)input_encryption_mode();
  const char* elementwise=std::getenv("CIPHEROPBENCH_ELEMENTWISE_ONLY");
  if(elementwise&&std::string(elementwise)!="1")throw std::invalid_argument("CIPHEROPBENCH_ELEMENTWISE_ONLY must be 1 or unset");
  if(elementwise&&with_bootstrap)throw std::invalid_argument("Elementwise-only keys cannot be used by bootstrapping or the benchmark runtime");
  const char* normalization=std::getenv("CIPHEROPBENCH_NORMALIZATION_ONLY");
  if(normalization&&(std::string(normalization)!="1"||with_bootstrap||elementwise))throw std::invalid_argument("Normalization-only keys require a non-bootstrap standalone check");
  reserve_gpu_memory();
  phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
  std::vector<int> bits{int(k::logq)};
  bits.insert(bits.end(), k::L, int(k::logp));
  bits.insert(bits.end(), k::boot_level, int(k::logq));
  bits.push_back(k::log_special_prime);
  parms.set_poly_modulus_degree(k::N);
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
  parms.set_secret_key_hamming_weight(unified_phantom::secret_hamming_weight);
#endif
  parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(k::N, bits));
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
  for(size_t i=0;i<parms.coeff_modulus().size();++i)
    if(parms.coeff_modulus()[i].value()!=unified_phantom::coefficient_primes[i])
      throw std::runtime_error("Runtime actual primes differ from the frozen common profile");
#endif
  auto context = std::make_shared<PhantomContext>(parms);
  auto secret = std::make_shared<PhantomSecretKey>(*context);
  auto pub = std::make_shared<PhantomPublicKey>(secret->gen_publickey(*context));
  auto relin = std::make_shared<PhantomRelinKey>(secret->gen_relinkey(*context));
  // Elementwise-only correctness checks omit unused rotation keys; HE parameters stay identical.
  std::vector<int> norm_steps{0,16384};for(int step=1;step<128;step*=2){norm_steps.push_back(step);norm_steps.push_back(-step);}
  const char* decoder=std::getenv("CIPHEROPBENCH_DECODER_ONLY");
  if(decoder&&(std::string(decoder)!="1"||with_bootstrap||elementwise||normalization))throw std::invalid_argument("Decoder-only keys require standalone non-bootstrap runtime");
  std::vector<int> decoder_steps{0};for(int step=1;step<32768;step*=4)decoder_steps.push_back(step);
  if(!rotation_steps.empty()&&(with_bootstrap||decoder||normalization||elementwise))
    throw std::invalid_argument("Custom rotation steps require a standalone non-bootstrap runtime");
  auto make_rotations=[&]{
    if(!rotation_steps.empty())return secret->create_galois_keys_from_steps(*context,rotation_steps);
    if(decoder)return secret->create_galois_keys_from_steps(*context,decoder_steps);
    if(normalization)return secret->create_galois_keys_from_steps(*context,norm_steps);
    if(elementwise)return secret->create_galois_keys_from_steps(*context,{0});
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
    return secret->create_galois_keys_from_steps(*context,unified_phantom::rotation_steps());
#else
    return secret->create_galois_keys(*context);
#endif
  };
  auto rotations=std::make_shared<PhantomGaloisKey>(make_rotations());
  if(normalization)std::cout<<"Key profile: standalone normalization, row/half rotations only, no bootstrap"<<std::endl;
  if(elementwise)std::cout<<"Key profile: elementwise-only, conjugation key; no bootstrap or matrix rotations"<<std::endl;
  auto encoder = std::make_shared<PhantomCKKSEncoder>(*context);
  ckks = std::make_shared<nexus::CKKSEvaluator>(context, pub, secret, encoder, relin, rotations, k::scale);
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
  ckks->set_logical_slots(unified_phantom::logical_slots);
#endif
  if (with_bootstrap) {
    // Upstream raw ModularReducer otherwise keeps the entire CKKS key context alive.
    bootstrapper = std::shared_ptr<::Bootstrapper>(new ::Bootstrapper(
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
      unified_phantom::bootstrap_log_width, unified_phantom::logical_log_slots, k::logN-1, k::total_level,
      k::scale, 25, unified_phantom::bootstrap_modular_degree, 2,
      unified_phantom::bootstrap_inverse_degree, ckks
#else
      10, k::logN-1, k::logN-1, k::total_level-1, k::scale, 25, 59, 2, 1, ckks
#endif
    ), [](::Bootstrapper* value){
      delete value->mod_reducer->poly_generator;
      delete value->mod_reducer->inverse_poly_generator;
      delete value->mod_reducer;
      delete value;
    });
    bootstrapper->prepare_mod_polynomial();
    std::vector<int> steps{0};
    for (int i=0; i<int(k::logN)-1; ++i) steps.push_back(1<<i);
    bootstrapper->addLeftRotKeys_Linear_to_vector_3(steps);
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
    bootstrapper->slot_vec.push_back(unified_phantom::logical_log_slots);
#else
    bootstrapper->slot_vec.push_back(k::logN-1);
#endif
    bootstrapper->generate_LT_coefficient_3();
  }
  synchronize();
}
void Runtime::synchronize() const {
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
Cipher Runtime::encrypt(k::FlatVec values, size_t chain) const {
 if(input_encryption_mode()=="public_key")return k::CKKSEncrypt(values,ckks,chain);
 auto plain=k::CKKSEncode(values,ckks);Cipher out;ckks->decryptor.encrypt_symmetric(plain,out);
 if(chain>1)ckks->evaluator.mod_switch_to_inplace(out,chain);return out;
}
CipherList Runtime::encrypt_hidden(torch::Tensor input, size_t chain) const {
  TORCH_CHECK(input.sizes() == torch::IntArrayRef({128,768}), "NEXUS hidden input requires [128,768]");
  CipherList out;
  for (auto& values : k::row_pack_128x768(input.to(torch::kDouble))) out.push_back(encrypt(values, chain));
  return out;
}
torch::Tensor Runtime::decrypt_slots(const Cipher& input) const {
  return k::tensor_from_vector(k::CKKSDecrypt(input, ckks), {2,128,128});
}
torch::Tensor Runtime::decrypt_hidden(const CipherList& input) const {
  TORCH_CHECK(input.size()==3, "hidden packing requires three ciphertexts");
  std::vector<torch::Tensor> blocks;
  for (const auto& ct : input) { auto t=decrypt_slots(ct); blocks.push_back(t[0]); blocks.push_back(t[1]); }
  return torch::cat(blocks, 1);
}
void Runtime::refresh(Cipher& input) {
  if (!bootstrapper) throw std::runtime_error("This level requires a bootstrapping runtime");
  Cipher before;if(refresh_observer)before=input;
  record("bootstrap", [&]{
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
    input=unified_phantom::precision_refresh(*ckks,*bootstrapper,input,false);
#else
    const char* precision=std::getenv("CIPHEROPBENCH_LEGACY_PRECISION_REFRESH");
    if(precision&&std::string(precision)!="1")throw std::invalid_argument("CIPHEROPBENCH_LEGACY_PRECISION_REFRESH must be 1 or unset");
    if(precision)detail::legacy_precision_refresh(input,bootstrapper);
    else k::bootstrap(input, bootstrapper, true);
#endif
  }, input.chain_index(), &input);
  if(refresh_observer)refresh_observer(before,input);
}
void Runtime::refresh(CipherList& input) { for(auto& ct : input) refresh(ct); }
void Runtime::record(const std::string& name, const std::function<void()>& fn, size_t in, const Cipher* out) {
  if (!tracing) { fn(); return; }
  synchronize();
  auto start = std::chrono::steady_clock::now();
  fn();
  synchronize();
  auto ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  events.push_back({name,ms,in,out ? out->chain_index() : 0});
  if(name=="batch.wo"||name=="batch.ln1"||name=="batch.ln2")
    std::cerr<<"Completed "<<name<<" in "<<ms/1000<<" s\n";
}
}
