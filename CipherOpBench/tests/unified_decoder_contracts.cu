#include "cipheropbench/decoder.cuh"
#include "cipheropbench/unified_phantom_parameters.cuh"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <complex>
#include <iomanip>
using namespace cipheropbench;
namespace u=cipheropbench::unified_phantom;
torch::Tensor signal(int rows,int width,double offset=0.){
  return torch::sin(torch::arange(rows*width,torch::kDouble)*.013).reshape({rows,width})*.1+offset;
}
int main(int argc,char**argv){try{
  TORCH_CHECK(argc==2,"Fresh output directory required");std::filesystem::path out=argv[1];
  torch::set_num_threads(2);torch::NoGradGuard guard;
  Runtime runtime(false);TORCH_CHECK(Runtime::input_encryption_mode()=="public_key","Public input required");
  auto client=runtime.ckks->decryptor;
  runtime.ckks->decryptor=nexus::Decryptor(runtime.ckks->context,nullptr);
  DecoderOperators ops(runtime);std::cout<<std::setprecision(17);
  auto dump=[&](const std::string& name,const DecoderRows& rows){
    std::filesystem::create_directories((out/name).parent_path());std::ofstream f(out/(name+".complex-f64"),std::ios::binary);
    for(const auto& ct:rows.blocks){
      TORCH_CHECK(ct.poly_modulus_degree()==(size_t(1)<<u::log_n)&&ct.scale()==u::scale,"Wrong HE geometry/scale");
      PhantomPlaintext pt;auto readable=ct;client.decrypt(readable,pt);std::vector<std::complex<double>> values;runtime.ckks->encoder.decode(pt,values);
      TORCH_CHECK(values.size()==u::physical_slots,"Partial physical readback");
      for(auto z:values){double pair[2]{z.real(),z.imag()};f.write(reinterpret_cast<char*>(pair),sizeof pair);}
    }
    TORCH_CHECK(f,"Incomplete readback file");
    std::cout<<"UNIFIED_DECODER_BUFFER {\"name\":\""<<name<<"\",\"rows\":"<<rows.rows<<",\"width\":"<<rows.width<<",\"stride\":"<<rows.stride<<",\"blocks\":"<<rows.blocks.size()<<",\"physical_slots\":"<<u::physical_slots<<"}"<<std::endl;
  };
  for(int width:{4096,1024}){
    auto input=ops.encrypt_rows(signal(9,width));for(const auto& ct:input.blocks)TORCH_CHECK(ct.coeff_modulus_size()==13,"Wrong input budget");
    auto result=ops.rope(input,4093);runtime.synchronize();
    auto name="rope-"+std::to_string(width);dump(name+"/output",result);dump(name+"/input-after",input);
  }
  EncryptedKVCache cache(ops,35,13);
  for(int count:{31,2,2}){
    int start=cache.length;auto key=ops.encrypt_rows(signal(count,1024,start*.01));
    auto value=ops.encrypt_rows(signal(count,1024,start*.01)*-.7);
    std::vector<std::pair<const void*,const void*>> stable;
    for(int block=0;block<cache.length/32;++block)stable.emplace_back(cache.key.blocks[block].data(),cache.value.blocks[block].data());
    cache.append(key,value,start);runtime.synchronize();
    for(size_t block=0;block<stable.size();++block)TORCH_CHECK(cache.key.blocks[block].data()==stable[block].first&&cache.value.blocks[block].data()==stable[block].second,"Stable cache history reallocated");
    for(const auto* rows:{&key,&value})for(const auto& ct:rows->blocks)TORCH_CHECK(ct.coeff_modulus_size()==13,"Caller input budget changed");
    for(const auto* rows:{&cache.key,&cache.value})for(const auto& ct:rows->blocks)TORCH_CHECK(ct.coeff_modulus_size()>=12&&ct.coeff_modulus_size()<=13,"Cached budget outside common ceiling");
    auto name="cache-"+std::to_string(start);dump(name+"/key-input-after",key);dump(name+"/value-input-after",value);dump(name+"/key",cache.key);dump(name+"/value",cache.value);
  }
  for(int position:{0,35}){
    bool rejected=false;try{auto extra=ops.encrypt_rows(signal(1,1024));cache.append(extra,extra,position);}catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected&&cache.length==35,"Cache overwrite/capacity guard failed");
  }
  EncryptedKVCache atomic(ops,35,13);auto past=ops.encrypt_rows(signal(31,1024));atomic.append(past,past,0);
  auto good=ops.encrypt_rows(signal(1,1024)),exhausted=good;
  runtime.ckks->evaluator.mod_switch_to_inplace(exhausted.blocks[0],k::chain_idx(1));
  bool rejected=false;try{atomic.append(good,exhausted,31);}catch(const std::exception&){rejected=true;}
  TORCH_CHECK(rejected&&atomic.length==31&&atomic.key.rows==31&&atomic.value.rows==31,"K/V preparation committed a partial failed append");
  dump("atomic/key",atomic.key);dump("atomic/value",atomic.value);
  std::ifstream maps("/proc/self/maps");std::ofstream mapped(out/"mapped-libraries.txt");mapped<<maps.rdbuf();
  std::cout<<"UNIFIED_DECODER_CONTRACT {\"native_server_private_key\":false,\"client_checks_between_appends\":true,\"cache_history_stable\":true,\"failed_append_atomic\":true,\"scope\":\"RoPE_and_KV_contracts_only\",\"benchmark_latency\":false}"<<std::endl;
  return 0;
}catch(const std::exception&e){std::cerr<<"UNIFIED_DECODER_CONTRACT_ERROR "<<e.what()<<std::endl;return 1;}}
