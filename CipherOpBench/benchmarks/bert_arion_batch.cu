#include "cipheropbench/column_methods.cuh"
#include "cipheropbench/research.cuh"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>

using namespace cipheropbench;

int main(int argc,char** argv){
 try{
  if(argc<6||argc>10)throw std::invalid_argument(
      "Usage: cipheropbench_bert_batch qk_head|qk|linear|block|encoder active warmup runs output [fixture-list] [profile] [arion|moai|euston|stip] [method-reference]");
  const std::string scope=argv[1];
  const std::string method=argc>=9?argv[8]:"arion";
  const bool stip=method=="stip";
  const bool diagnostic=method=="arion"&&std::getenv("CIPHEROPBENCH_ARION_STAGE_DIAGNOSTIC");
  const char* scalar=std::getenv("CIPHEROPBENCH_COLUMN_SCALAR_RNS");
  const int active=std::stoi(argv[2]),warmup=std::stoi(argv[3]),runs=std::stoi(argv[4]);
  TORCH_CHECK(active>=2&&active<=ColumnOperators::batch_capacity&&warmup>=0&&runs>0,"Invalid batch or repetition count");
  TORCH_CHECK(scope=="qk_head"||scope=="qk"||scope=="linear"||scope=="block"||scope=="encoder","Invalid benchmark scope");
  const bool qk=scope=="qk_head"||scope=="qk";
  const int head_count=scope=="qk"?12:1;
  const bool graph=scope=="block"||scope=="encoder";
  TORCH_CHECK(argc<10||graph,"A method reference requires block or encoder scope");
  TORCH_CHECK(method=="arion"||((method=="moai"||method=="euston"||stip)&&graph),"Expected a supported encoder method");
  TORCH_CHECK(!stip||(active==32&&argc==10),"STIP AMCP requires 32 samples and a validated method reference");
  const int layers=scope=="encoder"?12:1;
  torch::manual_seed(2026);
  std::vector<torch::Tensor> inputs,masks;
  Fixture fixture;
  const bool real=argc>=7;
  if(real){
   std::ifstream files(argv[6]);TORCH_CHECK(files.good(),"Cannot open fixture list");
   std::set<std::filesystem::path> paths;
   std::string line;
   for(int sample=0;sample<active;++sample){
    TORCH_CHECK(std::getline(files,line)&&!line.empty(),"Fixture list is shorter than active batch");
    auto path=std::filesystem::canonical(line);
    TORCH_CHECK(paths.insert(path).second,"Fixture list repeats an input path");
    auto sample_fixture=load_fixture(path,sample==0?layers:0);
    if(sample==0)fixture=sample_fixture;
    inputs.push_back(sample_fixture.hidden);
    masks.push_back(sample_fixture.mask);
   }
  }else{
   TORCH_CHECK(!graph,"Block and encoder require model traces");
   fixture=synthetic_fixture(1,2026);
   for(int sample=0;sample<active;++sample){
    inputs.push_back(torch::randn({128,768},torch::kDouble)*.10);
    masks.push_back(torch::ones({128,128},torch::kBool));
   }
  }
  auto hidden=torch::stack(inputs).to(torch::kDouble);
  auto mask=torch::stack(masks).select(1,0).to(torch::kBool);
  torch::Tensor method_reference;
  if(argc==10){
   auto values=calibration::read_reference(argv[9],size_t(active)*128*768);
   method_reference=torch::from_blob(values.data(),{active,128,768},torch::kDouble).clone();
  }
  std::vector<int> rotation_steps;
  if(scope=="linear")rotation_steps={0};
  if(qk){
   rotation_steps.push_back(16384);
   for(int step=ColumnOperators::batch_capacity;step<16384;step*=2){
    rotation_steps.push_back(step);rotation_steps.push_back(-step);
   }
  }
  Runtime runtime(graph,rotation_steps);
  if(method!="arion"&&!stip)runtime.calibration=calibration::load(argv[7],method);
  ColumnOperators operators(runtime,stip?"arion":method,method=="arion"&&argc>=8?argv[7]:"");
  std::optional<ResearchOperators> stip_ops;
  std::vector<StipBatchProfile> stip_profiles;
  std::array<CipherList,6> stip_first_projection;
  if(stip){
   std::ifstream profile(argv[7]);std::string magic,task,checkpoint,revision;int depth=0;
   profile>>magic>>task>>checkpoint>>revision>>depth;
   TORCH_CHECK(profile&&magic=="CIPHEROPBENCH_STIP_BATCH_PROFILE_V1"&&depth==12,"Invalid STIP batch profile");
   for(int i=0;i<depth;++i){
    StipBatchProfile row;profile>>row.norm1_divisor>>row.carry_divisor>>row.norm2_divisor;
    TORCH_CHECK(profile&&std::isfinite(row.norm1_divisor)&&row.norm1_divisor>0&&std::isfinite(row.carry_divisor)&&
                row.carry_divisor>=1&&std::isfinite(row.norm2_divisor)&&row.norm2_divisor>0,"Invalid STIP public scale");
    if(i<layers)stip_profiles.push_back(row);
   }
   TORCH_CHECK(!(profile>>magic),"Unexpected trailing STIP profile fields");
   stip_ops.emplace(runtime,"stip","",true);
   stip_first_projection=stip_ops->batch_stip_prepare(hidden,mask);
   std::cerr<<"STIP first-layer AMCP input encrypted\n";
  }
  std::vector<size_t> layer_ends;
  if(graph)runtime.layer_completed=[&](size_t completed,size_t count){
    if(runtime.tracing)layer_ends.push_back(runtime.events.size());
    std::cerr<<"Layers evaluated: "<<completed<<'/'<<count<<'\n';
  };
  std::vector<double> times(runs);
  std::map<std::string,std::vector<double>> stages;
  std::vector<std::map<std::string,std::vector<double>>> layer_stages(graph?layers:0);
  double max_error=0,max_error_vs_exact=0;
  bool valid=true;
  for(int head=0;head<head_count;++head){
   auto operand=hidden;
   torch::Tensor key_tensor,reference;
   if(qk){
    const auto& w=fixture.layers[0];
    auto project_head=[&](const LinearWeights& w){
     return (torch::matmul(hidden,w.weight.slice(1,head*64,(head+1)*64))+
             w.bias.slice(0,head*64,(head+1)*64)).contiguous();
    };
    key_tensor=project_head(w.key);
    operand=project_head(w.q);
    auto dense=torch::matmul(operand,key_tensor.transpose(1,2))*.125;
    reference=torch::empty_like(dense);
    for(int token=0;token<128;++token)
     reference.select(1,token).copy_(torch::roll(dense.select(1,token),-token,1));
   }else if(graph){
    std::vector<torch::Tensor> expected;
    for(int sample=0;sample<active;++sample){
     auto value=hidden[sample];
     for(const auto& w:fixture.layers)value=plain_block(value,masks[sample],w);
     expected.push_back(value);
    }
    reference=torch::stack(expected);
   }else reference=torch::matmul(hidden,fixture.layers[0].out.weight)+fixture.layers[0].out.bias;
   auto input=operators.encrypt_batch(operand,stip?k::chain_idx(8):k::boot_level+1);
   if(stip)std::cerr<<"STIP residual input encrypted\n";
   const auto key=qk?operators.encrypt_batch(key_tensor):CipherList{};
   for(int i=-warmup;i<runs;++i){
    auto x=stip&&i==runs-1?std::move(input):input;
    auto k=key;
    runtime.events.clear();
    layer_ends.clear();
    runtime.tracing=i>=0&&graph;
    runtime.synchronize();
    const auto start=std::chrono::steady_clock::now();
    CipherList output;
    if(stip)output=stip_ops->batch_stip_encoder(std::move(x),stip_first_projection,fixture.layers,mask,stip_profiles);
    else if(graph)output=operators.batch_encoder(std::move(x),fixture.layers,mask);
    else if(qk)output=operators.batch_qk(std::move(x),std::move(k));
    else output=operators.batch_linear(std::move(x),fixture.layers[0].out);
    runtime.synchronize();
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    runtime.tracing=false;
    if(i<0)continue;
    times[i]+=seconds;
    std::map<std::string,double> per_run;
    if(graph){
     TORCH_CHECK(layer_ends.size()==size_t(layers)&&layer_ends.back()==runtime.events.size(),"Incomplete layer timing boundaries");
     size_t begin=0;
     for(int layer=0;layer<layers;++layer){
      std::map<std::string,double> layer_run;
      for(size_t end=layer_ends[layer];begin<end;++begin){
       const auto& event=runtime.events[begin];
       if(event.name.starts_with("batch."))layer_run[event.name.substr(6)]+=event.milliseconds/1000;
      }
      for(const auto& [name,elapsed]:layer_run){
       layer_stages[layer][name].push_back(elapsed);
       per_run[name]+=elapsed;
      }
     }
    }
    for(const auto& [name,total]:per_run)stages[name].push_back(total);
    const auto actual=operators.decrypt_batch(output,active);
    for(int sample=0;sample<active;++sample){
     auto exact=reference[sample],observed=actual[sample];
     auto expected=method_reference.defined()?method_reference[sample]:exact;
     if(graph){
      const auto keep=mask[sample].unsqueeze(-1).expand_as(expected);
      exact=exact.masked_select(keep);expected=expected.masked_select(keep);observed=observed.masked_select(keep);
     }
     const bool finite=torch::isfinite(observed).all().item<bool>();
     valid=valid&&finite&&torch::allclose(observed,expected,1e-3,graph?1e-2:1e-4);
     if(finite){
      max_error=std::max(max_error,(observed-expected).abs().max().item<double>());
      max_error_vs_exact=std::max(max_error_vs_exact,(observed-exact).abs().max().item<double>());
     }
    }
   }
   if(qk)std::cerr<<"Validated through head "<<head+1<<'/'<<head_count<<": "<<valid<<'\n';
  }
  std::ofstream file(argv[5],std::ios::app);TORCH_CHECK(file.good(),"Cannot open result file");
  file<<std::setprecision(14)<<"{\"method\":\""<<method<<"\",\"scope\":\""<<scope
      <<"\",\"implementation\":\"cipheropbench_common_library\",\"backend\":\"PhantomBootstrapping\",\"layout\":\""<<(stip?"stip_amcp_dhp_rihp":"token_sample_interleaved")
      <<"\",\"active_samples\":"<<active
      <<",\"physical_batch_capacity\":"<<(stip?32:ColumnOperators::batch_capacity)
      <<",\"attention_heads\":"<<(qk?head_count:12)
      <<",\"executed_layers\":"<<layers<<",\"sequence_length\":128,\"error_samples\":"<<active
      <<",\"evidence\":\""<<(real?"model_trace":"synthetic")<<"\",\"status\":\""
      <<(diagnostic?"diagnostic_only":valid?"validated":"numerical_mismatch")<<"\",\"warmup\":"<<warmup<<",\"runs\":"<<runs
      <<",\"reference_kind\":\""<<(method_reference.defined()?"noiseless_cipheropbench_method_circuit":"exact_transformer")
      <<"\",\"projection_kernel\":\""<<(scalar&&std::string(scalar)=="1"?"scalar_rns":"encoded_constant")
      <<"\",\"refresh_profile\":\""<<(std::getenv("CIPHEROPBENCH_LEGACY_PRECISION_REFRESH")?"legacy_precision":"legacy_default")
      <<"\",\"arion_gain_profile\":\""<<(std::getenv("CIPHEROPBENCH_ARION_GAIN_FOLDED")?"folded_into_mask":"default")
      <<"\",\"arion_refresh_profile\":\""<<(std::getenv("CIPHEROPBENCH_ARION_REFRESH_PROFILE")?"training_head_bounds":"default")
      <<"\",\"max_error\":"<<max_error<<",\"max_error_vs_exact\":"<<max_error_vs_exact<<",\"batch_seconds\":[";
  auto emit=[&](const std::vector<double>& data){
   for(size_t j=0;j<data.size();++j){if(j)file<<',';file<<data[j];}
  };
  if(!diagnostic)emit(times);
  auto emit_stages=[&](const auto& group){
   file<<'{';bool comma=false;
   for(const auto& [name,data]:group){if(comma)file<<',';file<<'"'<<name<<"\":[";emit(data);file<<']';comma=true;}
   file<<'}';
  };
  file<<"],\"operator_batch_seconds\":";emit_stages(stages);
  if(graph){
   file<<",\"layer_operator_batch_seconds\":[";
   for(int i=0;i<layers;++i){if(i)file<<',';emit_stages(layer_stages[i]);}
   file<<']';
  }
  file<<"}\n";
  return valid?0:2;
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
