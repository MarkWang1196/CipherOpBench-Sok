#include "cipheropbench/cachemir_gpu.cuh"
#include "silu_coefficients.cuh"
#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cuda_runtime_api.h>
using namespace cipheropbench::cachemir;
void save(const std::string& path,const std::vector<std::complex<double>>& x){
  std::ofstream f(path,std::ios::binary);for(auto z:x){double a[2]{z.real(),z.imag()};f.write(reinterpret_cast<char*>(a),16);}
  if(!f)throw std::runtime_error("Incomplete vector write");
}
int main(int argc,char** argv){try{
  if(argc!=2)throw std::invalid_argument("Output directory required");std::string out=argv[1];
  std::cout<<std::setprecision(17);
  size_t free_memory = 0, total_memory = 0;
  if (cudaMemGetInfo(&free_memory, &total_memory) != cudaSuccess || free_memory < (72ULL << 30))
    throw std::runtime_error("Cachemir benchmark requires at least 72 GiB of free GPU memory");
  phantom::EncryptionParameters p(phantom::scheme_type::ckks);p.set_poly_modulus_degree(size_t(1)<<cipheropbench::unified_phantom::log_n);
  std::vector<int> bits{cipheropbench::unified_phantom::log_q0};bits.insert(bits.end(),cipheropbench::unified_phantom::data_primes,cipheropbench::unified_phantom::log_scale);bits.insert(bits.end(),cipheropbench::unified_phantom::bootstrap_primes+1,cipheropbench::unified_phantom::log_bootstrap_prime);
  p.set_secret_key_hamming_weight(cipheropbench::unified_phantom::secret_hamming_weight);
  p.set_coeff_modulus(phantom::arith::CoeffModulus::Create(size_t(1)<<cipheropbench::unified_phantom::log_n,bits));
  auto context=std::make_shared<PhantomContext>(p);
  auto sk=std::make_shared<PhantomSecretKey>(*context);
  auto pk=std::make_shared<PhantomPublicKey>(sk->gen_publickey(*context));
  auto rk=std::make_shared<PhantomRelinKey>(sk->gen_relinkey(*context));
  auto gk=std::make_shared<PhantomGaloisKey>(sk->create_galois_keys_from_steps(*context,cipheropbench::unified_phantom::rotation_steps()));
  auto encoder=std::make_shared<PhantomCKKSEncoder>(*context);
  nexus::CKKSEvaluator client(context,pk,sk,encoder,rk,gk,scale);
  client.set_logical_slots(slots);
  GPU server(context,pk,encoder,rk,gk,true);
  Values input(slots,0),gamma(hidden);for(int i=0;i<hidden;++i){input[i*8]=.3+.08*std::sin((i+1)*.017);gamma[i]=1+.03*std::cos((i+1)*.031);}
  PhantomPlaintext pt;client.encoder.encode(input,scale,pt);Cipher encrypted;client.encryptor.encrypt(pt,encrypted);
  client.evaluator.mod_switch_to_inplace(encrypted,cipheropbench::unified_phantom::input_chain_index);GPU::synchronize();
  auto decode=[&](const std::string& name,Cipher x){client.decryptor.decrypt(x,pt);std::vector<std::complex<double>> v;client.encoder.decode(pt,v);if(v.size()!=cipheropbench::unified_phantom::physical_slots)throw std::runtime_error("Incomplete readback");save(out+"/"+name+".complex-f64",v);};
  // Client verification is after each complete component, outside its timer.
  // No decoded vector or client callback is supplied to server operations.
  auto run=[&](const std::string& name,const std::function<Cipher()>& fn){
    GPU::synchronize();auto begin=std::chrono::steady_clock::now();Cipher y=fn();GPU::synchronize();
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<"CACHEMIR_GPU_COMPONENT {\"stage\":\""<<name<<"\",\"server_online_ms\":"<<ms<<",\"output_moduli\":"<<y.coeff_modulus_size()<<",\"output_log_scale\":"<<std::log2(y.scale())<<",\"server_has_private_key\":false,\"full_decoder\":false,\"comparison_eligible\":false}"<<std::endl;
    decode(name,std::move(y));
  };
  decode("input",encrypted);
  run("rms",[&]{return server.rms(encrypted,gamma);});
  run("rope",[&]{return server.rope(encrypted,7);});
  run("linear",[&]{return server.linear(encrypted,[](int r,int c){return (r==c?.5:0.)+(((r*131+c*17)%257)-128)*.000002;});});
  auto coefficients=release_silu_coefficients();run("silu",[&]{return server.silu(encrypted,coefficients);});
  auto public_evaluator=std::make_shared<nexus::CKKSEvaluator>(context,pk,nullptr,encoder,rk,gk,scale);
  public_evaluator->set_logical_slots(slots);
  NTL::RR even_coefficients[9];for(auto& a:even_coefficients)a=NTL::to_RR(0);
  even_coefficients[0]=NTL::to_RR(.25);even_coefficients[2]=NTL::to_RR(.4);even_coefficients[8]=NTL::to_RR(.02);
  boot::Polynomial even(8,even_coefficients,"cheb");even.generate_poly_heap();
  run("even_polynomial",[&]{Cipher result;even.homomorphic_poly_evaluation(public_evaluator,result,encrypted);return result;});
  run("refresh_sparse",[&]{return server.refresh(encrypted,2.,13,hidden);});
  run("refresh_deep_rms",[&]{auto y=server.rms(encrypted,gamma,.5);return server.refresh_scaled(std::move(y),2.,13,hidden);});
  Values dense(slots);for(int i=0;i<slots;++i)dense[i]=.8+.1*std::sin(i*.017);
  Cipher dense_ct;client.encoder.encode(dense,scale,pt);client.encryptor.encrypt(pt,dense_ct);client.evaluator.mod_switch_to_inplace(dense_ct,cipheropbench::unified_phantom::input_chain_index);
  run("refresh_dense",[&]{return server.refresh(dense_ct,2.,13);});
  run("rotate_negative3",[&]{Cipher y;public_evaluator->evaluator.rotate_vector(encrypted,-3,*gk,y);return y;});
  std::vector<cuDoubleComplex> device_values(slots);for(int i=0;i<slots;++i)device_values[i]=make_cuDoubleComplex(input[i],0);
  auto device=phantom::util::make_cuda_auto_ptr<cuDoubleComplex>(slots,nullptr);
  auto status=cudaMemcpy(device.get(),device_values.data(),slots*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
  Cipher device_ct;client.encoder.encode(device,scale,pt);client.encryptor.encrypt(pt,device_ct);client.evaluator.mod_switch_to_inplace(device_ct,cipheropbench::unified_phantom::input_chain_index);
  run("device_encoder_roundtrip",[&]{return device_ct;});
  decode("input-after",encrypted);
  std::cout<<"CACHEMIR_GPU_END {\"status\":\"encrypted_execution_complete_pending_independent_gates\",\"maximum_product_scale_gain_error\":"<<server.maximum_product_scale_gain_error()<<",\"N\":131072,\"hidden\":4096,\"source_square_diagonals\":512,\"full_decoder\":false,\"comparison_eligible\":false}"<<std::endl;
  std::ifstream maps("/proc/self/maps");std::ofstream mapped(out+"/mapped-libraries.txt");mapped<<maps.rdbuf();return 0;
}catch(const std::exception& e){std::cerr<<"CACHEMIR_GPU_ERROR "<<e.what()<<std::endl;return 1;}}
