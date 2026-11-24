#include "cipheropbench/decoder.cuh"
#include "cipheropbench/unified_phantom_parameters.cuh"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <complex>
using namespace cipheropbench;
namespace u=cipheropbench::unified_phantom;
torch::Tensor signal(int rows,int width,double amplitude,double offset){return torch::sin(torch::arange(rows*width,torch::kDouble)*.013).reshape({rows,width})*amplitude+offset;}
struct Step{int start,count;DecoderRows q,k,v,qr,kr,score,p,output,cache_k,cache_v;};
int main(int argc,char**argv){try{
 TORCH_CHECK(argc==2,"Fresh output directory required");std::filesystem::path directory=argv[1];torch::set_num_threads(2);torch::NoGradGuard guard;
 Runtime runtime(true);TORCH_CHECK(Runtime::input_encryption_mode()=="public_key","Public input required");
 auto client=runtime.ckks->decryptor;runtime.ckks->decryptor=nexus::Decryptor(runtime.ckks->context,nullptr);
 DecoderOperators ops(runtime);EncryptedKVCache cache(ops,16,13);std::vector<Step> steps;
 for(int count:{8,1,1}){
  Step step;step.start=cache.length;step.count=count;
  step.q=ops.encrypt_rows(signal(count,4096,.05,step.start*.003));step.k=ops.encrypt_rows(signal(count,1024,.04,step.start*.002));step.v=ops.encrypt_rows(signal(count,1024,.2,step.start*.01));
  runtime.events.clear();runtime.tracing=true;
  step.qr=ops.rope(step.q,step.start);step.kr=ops.rope(step.k,step.start);cache.append(step.kr,step.v,step.start);
  step.score=ops.qk(step.qr,cache.key,step.start);step.p=ops.softmax(step.score,count,step.start,cache.length);step.output=ops.av(step.p,cache.value,count);runtime.synchronize();
  TORCH_CHECK(runtime.events.size()==3,"Unexpected attention refresh schedule");for(const auto&e:runtime.events)TORCH_CHECK(e.name=="decoder_public_bounded_restore","Implicit unbounded attention refresh");
  for(const auto&c:step.p.blocks)TORCH_CHECK(c.coeff_modulus_size()==12,"Probability budget");
  step.cache_k=cache.key;step.cache_v=cache.value;steps.push_back(std::move(step));
  std::cout<<"UNIFIED_GQA_STEP {\"start\":"<<steps.back().start<<",\"query_tokens\":"<<count<<",\"cache_tokens\":"<<cache.length<<",\"bounded_restore_calls\":3,\"shared_precision_refresh_calls\":6,\"physical_bootstrap_calls\":12,\"client_decrypt_between\":false}"<<std::endl;
 }
 // No client decryption occurs until all three connected attention forwards finish.
 auto dump=[&](const std::string&name,const DecoderRows&rows){
  std::filesystem::create_directories((directory/name).parent_path());std::ofstream f(directory/(name+".complex-f64"),std::ios::binary);
  for(auto ct:rows.blocks){PhantomPlaintext pt;client.decrypt(ct,pt);std::vector<std::complex<double>> values;runtime.ckks->encoder.decode(pt,values);TORCH_CHECK(values.size()==u::physical_slots,"Partial physical readback");for(auto z:values){double pair[2]{z.real(),z.imag()};f.write(reinterpret_cast<char*>(pair),sizeof pair);}}
  TORCH_CHECK(f,"Incomplete raw readback");std::cout<<"UNIFIED_GQA_BUFFER {\"name\":\""<<name<<"\",\"rows\":"<<rows.rows<<",\"width\":"<<rows.width<<",\"stride\":"<<rows.stride<<",\"blocks\":"<<rows.blocks.size()<<"}"<<std::endl;
 };
 for(size_t i=0;i<steps.size();++i){const auto&s=steps[i];std::string p="step-"+std::to_string(i)+"/";
  dump(p+"q-input-after",s.q);dump(p+"k-input-after",s.k);dump(p+"v-input-after",s.v);dump(p+"rope-q",s.qr);dump(p+"rope-k",s.kr);dump(p+"scores",s.score);dump(p+"probabilities",s.p);dump(p+"av",s.output);dump(p+"cache-k",s.cache_k);dump(p+"cache-v",s.cache_v);
 }
 std::ifstream maps("/proc/self/maps");std::ofstream mapped(directory/"mapped-libraries.txt");mapped<<maps.rdbuf();
 std::cout<<"UNIFIED_GQA_CONTRACT {\"native_server_private_key\":false,\"query_heads\":32,\"kv_heads\":8,\"head_dim\":128,\"client_decrypt_between\":false,\"QKV_client_fixture\":true,\"QKV_O_FFN_connected\":false,\"benchmark_latency\":false}"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<"UNIFIED_GQA_ERROR "<<e.what()<<std::endl;return 1;}}
