#include "cipheropbench/cachemir_gpu.cuh"
#include "silu_coefficients.cuh"
#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <cuda_runtime_api.h>
using namespace cipheropbench::cachemir;
double dense_weight(int row,int col,int seed){
  uint64_t x=uint64_t(row+1)*0x9e3779b97f4a7c15ULL ^ uint64_t(col+1)*0xbf58476d1ce4e5b9ULL ^ uint64_t(seed)*0x94d049bb133111ebULL;
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31;
  return (double(x&65535)/65535-.5)*(seed<=3?.0008:.02);
}
void save(const std::string& path,const std::vector<std::complex<double>>& x){
  auto p=std::filesystem::path(path);std::filesystem::create_directories(p.parent_path());
  std::ofstream f(path,std::ios::binary);for(auto z:x){double a[2]{z.real(),z.imag()};f.write(reinterpret_cast<char*>(a),16);}
  if(!f)throw std::runtime_error("Incomplete vector write");
}
int main(int argc,char** argv){try{
  if(argc!=2)throw std::invalid_argument("Output directory required");std::string out=argv[1];std::cout<<std::setprecision(17);
  size_t free_memory = 0, total_memory = 0;
  if (cudaMemGetInfo(&free_memory, &total_memory) != cudaSuccess || free_memory < (72ULL << 30))
    throw std::runtime_error("Cachemir benchmark requires at least 72 GiB of free GPU memory");
  phantom::EncryptionParameters p(phantom::scheme_type::ckks);p.set_poly_modulus_degree(size_t(1)<<cipheropbench::unified_phantom::log_n);
  std::vector<int> bits{cipheropbench::unified_phantom::log_q0};bits.insert(bits.end(),cipheropbench::unified_phantom::data_primes,cipheropbench::unified_phantom::log_scale);bits.insert(bits.end(),cipheropbench::unified_phantom::bootstrap_primes+1,cipheropbench::unified_phantom::log_bootstrap_prime);
  p.set_coeff_modulus(phantom::arith::CoeffModulus::Create(size_t(1)<<cipheropbench::unified_phantom::log_n,bits));p.set_secret_key_hamming_weight(cipheropbench::unified_phantom::secret_hamming_weight);
  auto context=std::make_shared<PhantomContext>(p);std::cout<<"CACHEMIR_LAYER_STAGE generate_public_keys"<<std::endl;
  auto sk=std::make_shared<PhantomSecretKey>(*context);auto pk=std::make_shared<PhantomPublicKey>(sk->gen_publickey(*context));
  auto rk=std::make_shared<PhantomRelinKey>(sk->gen_relinkey(*context));auto gk=std::make_shared<PhantomGaloisKey>(sk->create_galois_keys_from_steps(*context,cipheropbench::unified_phantom::rotation_steps()));
  auto encoder=std::make_shared<PhantomCKKSEncoder>(*context);nexus::CKKSEvaluator client(context,pk,sk,encoder,rk,gk,scale);
  client.set_logical_slots(slots);
  GPU server(context,pk,encoder,rk,gk,true);LayerWeights weights;
  weights.q=[](int row,int col){int h=row%32,d=row/32;return dense_weight(h*128+d,col,1);};
  weights.k=[](int row,int col){int h=row%32,d=row/32;return h%4?0.:dense_weight((h/4)*128+d,col,2);};
  weights.v=[](int row,int col){int h=row%32,d=row/32;return h%4?0.:dense_weight((h/4)*128+d,col,3);};
  weights.o=[](int row,int col){return dense_weight(row,col,4);};weights.gate=[](int row,int col){return dense_weight(row,col,5);};
  weights.up=[](int row,int col){return dense_weight(row,col,6);};weights.down=[](int row,int col){return dense_weight(row,col,7);};
  for(int f=0;f<hidden;++f){weights.gamma_attention.push_back(1+.03*std::cos((f+1)*.031));weights.gamma_ffn.push_back(1+.02*std::sin((f+1)*.023));}
  weights.silu_coefficients=release_silu_coefficients();Layer layer(server,std::move(weights));
  PhantomPlaintext pt;std::vector<Cipher> encrypted;
  for(int position=0;position<2;++position){Values input(slots,0);for(int f=0;f<hidden;++f)input[f*8]=.3+.08*std::sin((f+1)*.017+position*.19);
    client.encoder.encode(input,scale,pt);Cipher ct;client.encryptor.encrypt(pt,ct);client.evaluator.mod_switch_to_inplace(ct,cipheropbench::unified_phantom::input_chain_index);encrypted.push_back(std::move(ct));}
  auto decode=[&](const std::string& name,Cipher x){client.decryptor.decrypt(x,pt);std::vector<std::complex<double>> values;client.encoder.decode(pt,values);if(values.size()!=cipheropbench::unified_phantom::physical_slots)throw std::runtime_error("Incomplete readback");save(out+"/"+name+".complex-f64",values);};
  std::vector<LayerResult> results;std::vector<Cache> snapshots;
  for(int position=0;position<2;++position){
    std::cout<<"CACHEMIR_LAYER_STAGE token="<<position<<" encrypted_forward"<<std::endl;GPU::synchronize();auto start=std::chrono::steady_clock::now();
    auto result=layer.apply(encrypted[position],position);GPU::synchronize();double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"CACHEMIR_GPU_LAYER {\"position\":"<<position<<",\"tokens_in_encrypted_KV\":"<<layer.cache().next<<",\"server_online_ms\":"<<ms<<",\"bootstrap_calls\":"<<result.bootstrap_calls<<",\"physical_bootstrap_calls\":"<<result.bootstrap_calls*cipheropbench::unified_phantom::physical_bootstraps_per_refresh<<",\"output_moduli\":"<<result.output.coeff_modulus_size()<<",\"server_has_private_key\":false,\"intermediate_client_decryptions\":0,\"hidden\":4096,\"intermediate\":14336,\"q_heads\":32,\"kv_heads\":8,\"head_dimension\":128,\"full_layer_connected\":true,\"validation_status\":\"pending_independent_full_slot_gates\",\"comparison_eligible\":false,\"stage_ms\":{";
    bool first=true;for(const auto& entry:result.stage_ms){if(!first)std::cout<<",";first=false;std::cout<<"\""<<entry.first<<"\":"<<entry.second;}std::cout<<"}}"<<std::endl;
    results.push_back(std::move(result));snapshots.push_back(layer.cache());
  }
  for(int position=0;position<2;++position){const auto& result=results.at(position);const auto& snapshot=snapshots.at(position);
    // All server operations have finished before the client reads any trace.
    std::string prefix="token-"+std::to_string(position)+"/";
    decode(prefix+"input-after",encrypted[position]);for(const auto& entry:result.trace)decode(prefix+entry.first,entry.second);
    for(const auto& entry:snapshot.pages){int page=entry.first;for(const auto& block:entry.second->k)decode(prefix+"cache-k/"+std::to_string(page)+"/"+std::to_string(block.first),*block.second);
      for(int i=0;i<128;++i)decode(prefix+"cache-v/"+std::to_string(page)+"/"+std::to_string(i),*entry.second->v[i]);}
  }
  std::ifstream maps("/proc/self/maps");std::ofstream mapped(out+"/mapped-libraries.txt");mapped<<maps.rdbuf();return 0;
}catch(const std::exception& e){std::cerr<<"CACHEMIR_GPU_LAYER_ERROR "<<e.what()<<std::endl;return 1;}}
