#include "cipheropbench/components.cuh"
#include "cipheropbench/component_plans.hpp"
#include "cipheropbench/method_registry.cuh"
#include "literature_math.cuh"
namespace cipheropbench {
namespace {
using namespace detail;
using component::Complex;
int hidden_index(int row,int feature){return (feature%256/128)*half+row*128+feature%128;}
Cipher complex_plain(Math& m,Cipher x,const std::vector<Complex>& values){
 TORCH_CHECK(values.size()==slots,"Complex plaintext has wrong slot count");PhantomPlaintext p;k::FlatVec v(slots);for(int j=0;j<slots;++j)v[j]=make_cuDoubleComplex(values[j].real(),values[j].imag());
 double prime=m.r.ckks->context->get_context_data(x.chain_index()).parms().coeff_modulus().back().value();m.r.ckks->encoder.encode(v,x.chain_index(),prime*k::scale/x.scale(),p);m.r.ckks->evaluator.multiply_plain_inplace(x,p);m.r.ckks->evaluator.rescale_to_next_inplace(x);x.set_scale(k::scale);return x;
}
Cipher fourier(Math& m,Cipher x,int degree,bool inverse){
 static std::map<std::pair<int,bool>,std::vector<component::FourierStage>> cache;auto key=std::pair{degree,inverse};if(!cache.count(key))cache[key]=component::fourier_stages(degree,inverse);
 for(const auto& stage:cache.at(key)){TORCH_CHECK(x.coeff_modulus_size()>1,"Polynomial transform ran out of levels; complex intermediates cannot use the real bootstrap");Cipher y;bool first=true;
  for(const auto& [shift,d]:stage){std::vector<Complex> v(slots);for(int j=0;j<slots;++j)v[j]=d[j%degree];auto term=complex_plain(m,m.rot(x,shift),v);y=first?std::move(term):m.add(y,term);first=false;}x=std::move(y);
 }return x;
}
std::vector<Complex> weight_evaluations(torch::Tensor w,int input,int output,const component::PolynomialPlan& p){
 std::vector<Complex> x(p.degree);auto values=w.accessor<double,2>();
 for(int o=0;o<p.output&&output+o<w.size(1);++o)for(int i=0;i<p.input&&input+i<w.size(0);++i){int j=o*p.input+p.input-i-1;x[j]=values[input+i][output+o]*std::polar(1.,-std::acos(-1.)*j/p.degree);}
 // Unnormalized DIF: evaluation at the odd roots, in bit-reversed order.
 for(int length=p.degree;length>=2;length/=2)for(int base=0;base<p.degree;base+=length)for(int j=0;j<length/2;++j){auto a=x[base+j],b=x[base+j+length/2];x[base+j]=a+b;x[base+j+length/2]=(a-b)*std::polar(1.,-2*std::acos(-1.)*j/length);}
 std::vector<Complex> v(slots);for(int j=0;j<slots;++j)v[j]=x[j%p.degree];return v;
}
CipherList polynomial_linear(Math& m,CipherList x,const LinearWeights& w){
 // Realize R[X]/(X^32768+1) in CKKS evaluation slots at the common N=65536.
 // FFT adapters are charged; no BFV modular arithmetic or MPC traffic is claimed.
 auto p=component::polynomial_plan(128,768,768,slots);TORCH_CHECK(p.batch==128&&p.input==16&&p.output==16,"Unexpected common polynomial block plan");
 for(auto& c:x)m.need(c,14);CipherList evaluations;auto weights=w.weight.contiguous();
 for(int input=0;input<768;input+=p.input){auto packed=m.route(x,[=](int j){int row=j/(p.input*p.output),i=j%(p.input*p.output);if(row>=128||i>=p.input)return std::tuple{-1,0,0.};int f=input+i;return std::tuple{f/256,hidden_index(row,f),1.};});evaluations.push_back(fourier(m,std::move(packed),p.degree,false));}
 CipherList out(3);bool initialized[3]{false,false,false};
 for(int output=0;output<768;output+=p.output){Cipher sum;
  for(int input=0;input<768;input+=p.input){auto term=complex_plain(m,evaluations[input/p.input],weight_evaluations(weights,input,output,p));sum=input?m.add(sum,term):std::move(term);}
  sum=fourier(m,std::move(sum),p.degree,true);
  // First gather within each token, then place tokens in the canonical row layout.
  auto gathered=m.route({sum},[=](int j){int row=j/(p.input*p.output),o=j%(p.input*p.output);if(row>=128||o>=p.output)return std::tuple{-1,0,0.};return std::tuple{0,row*p.input*p.output+o*p.input+p.input-1,1.};});
  int group=output/256;auto packed=m.route({gathered},[=](int j){int row=(j%half)/128,f=group*256+(j/half)*128+j%128;if(f<output||f>=output+p.output)return std::tuple{-1,0,0.};return std::tuple{0,row*p.input*p.output+f-output,1.};});
  out[group]=initialized[group]?m.add(out[group],packed):std::move(packed);initialized[group]=true;
 }auto biases=k::row_pack_768x1(w.bias);for(int g=0;g<3;++g){auto pt=k::CKKSEncode(biases[g],m.r.ckks,&out[g]);m.r.ckks->evaluator.add_plain_inplace(out[g],pt);}return out;
}
CipherList imra_linear(Math& m,CipherList x,const LinearWeights& w){
 constexpr int nd=256,kd=256,lane=1024,batch=32;const auto plan=component::imra_plan(nd,kd,lane);int bs=plan.baby,ps=plan.horizontal,gs=plan.giants(),sf=gs*bs;
 for(auto& c:x)m.need(c,4);auto weights=w.weight.contiguous();auto a=weights.accessor<double,2>();CipherList out(3);bool initialized[3]{false,false,false};
 for(int token=0;token<128;token+=batch){std::vector<CipherList> babies(3);
  for(int input=0;input<3;++input){auto repeated=m.route(x,[=](int j){int row=token+j/lane,f=input*kd+j%kd;return std::tuple{f/256,hidden_index(row,f),1.};});for(int b=0;b<bs;++b)babies[input].push_back(m.rot(repeated,b));}
  for(int output=0;output<3;++output){Cipher result;bool first=true;
   for(int g=0;g<gs;++g){Cipher bucket;bool empty=true;
    for(int input=0;input<3;++input)for(int b=0;b<bs;++b){Vec v(slots);bool nonzero=false;
     for(int j=0;j<lane;++j){int p=j/(kd+sf),local=j%(kd+sf);if(p>=ps)continue;int in=(local+p*sf+b)%kd,of=(local-g*bs+kd)%kd;double value=a[input*kd+in][output*nd+of];nonzero|=value!=0.;for(int t=0;t<batch;++t)v[t*lane+j]=value;}
     if(!nonzero)continue;auto term=m.pm(babies[input][b],v);bucket=empty?std::move(term):m.add(bucket,term);empty=false;
    }if(empty)continue;bucket=m.rot(bucket,g*bs);result=first?std::move(bucket):m.add(result,bucket);first=false;
   }if(first)result=m.zero(babies[0][0]);
   auto combined=result;for(int p=1;p<ps;++p)combined=m.add(combined,m.rot(result,p*(kd+sf)));
   auto packed=m.route({combined},[=](int j){int row=(j%half)/128;if(row<token||row>=token+batch)return std::tuple{-1,0,0.};int f=(j/half)*128+j%128;return std::tuple{0,(row-token)*lane+f,1.};});
   out[output]=initialized[output]?m.add(out[output],packed):std::move(packed);initialized[output]=true;
  }
 }auto bias=k::row_pack_768x1(w.bias);for(int g=0;g<3;++g){auto p=k::CKKSEncode(bias[g],m.r.ckks,&out[g]);m.r.ckks->evaluator.add_plain_inplace(out[g],p);}return out;
}
}
bool component_scope_supported(const std::string& method,const std::string& scope){return !is_component_method(method)||scope=="linear"||scope=="qkv"||scope=="bootstrap"||scope=="ckks_add"||scope=="ckks_mul";}
ComponentOperators::ComponentOperators(Runtime& r,std::string name):TransformerOperators(r),method_(std::move(name)){TORCH_CHECK(is_component_method(method_),"Unknown FHE component method");TORCH_CHECK(!r.calibration.enabled,"Component methods do not define nonlinear calibration");}
CipherList ComponentOperators::linear(CipherList x,const LinearWeights& w){TORCH_CHECK(x.size()==3&&w.weight.sizes()==torch::IntArrayRef({768,768})&&w.bias.numel()==768,"Component linear requires the common 128x768 @ 768x768 shape");Math m{runtime_};m.prime_scale=true;CipherList out;runtime_.record(method_=="private_llm"?"private_llm_negacyclic_polynomial_with_fft_adapters":"metakernel_imra_horizontal_vertical_with_adapters",[&]{out=method_=="private_llm"?polynomial_linear(m,std::move(x),w):imra_linear(m,std::move(x),w);});return out;}
QKV ComponentOperators::project_qkv(CipherList x,const LayerWeights& w){Math m{runtime_};m.prime_scale=true;auto q=linear(x,w.q),key=linear(x,w.key),v=linear(std::move(x),w.v);QKV out;constexpr std::array<std::array<int,2>,6> pairs{{{0,3},{2,1},{4,7},{6,5},{8,11},{10,9}}};
 for(int p=0;p<6;++p){auto [a,b]=pairs[p];auto route=[=](int j){int h=j/half,row=(j%half)/128,c=j%128;if(c>=64)return std::tuple{-1,0,0.};int f=(h?b:a)*64+c;return std::tuple{f/256,hidden_index(row,f),1.};};out.q[p]=m.route(q,route);out.key[p]=m.route(key,route);out.v[p]=m.route(v,[=](int j){int row=(j%half)/128,c=j%128,f=(c<64?a:b)*64+c%64;return std::tuple{f/256,hidden_index(row,f),1.};});}return out;
}
[[noreturn]] void ComponentOperators::unsupported()const{TORCH_CHECK(false,method_," exposes FHE plaintext-weight matrix components only; nonlinear/MPC protocols and a complete encoder are not implemented");}
Cipher ComponentOperators::qk(Cipher,Cipher){unsupported();}Cipher ComponentOperators::av(Cipher,Cipher){unsupported();}Cipher ComponentOperators::softmax(Cipher,const k::FlatVec&){unsupported();}Cipher ComponentOperators::softmax_v(Cipher,Cipher,const k::FlatVec&){unsupported();}
CipherList ComponentOperators::attention(CipherList,const LayerWeights&,torch::Tensor){unsupported();}CipherList ComponentOperators::gelu(CipherList){unsupported();}CipherList ComponentOperators::layer_norm(CipherList,torch::Tensor,torch::Tensor){unsupported();}CipherList ComponentOperators::ffn(CipherList,const LayerWeights&){unsupported();}CipherList ComponentOperators::block(CipherList,const LayerWeights&,torch::Tensor){unsupported();}CipherList ComponentOperators::encoder(CipherList,const std::vector<LayerWeights>&,torch::Tensor){unsupported();}
void register_component_methods(){for(auto name:{"private_llm","metakernel"})MethodRegistry::instance().add({name,"PhantomBootstrapping_common",std::string(name)=="private_llm"?"private_llm_polynomial_ckks_fft_adapter":"metakernel_imra_bs8_ps2_token32",std::string(name)=="private_llm"?"3baffb3aaac0748107c56e026fd63c348e317d17":"d0c14ab101c1f1fedfb31d8fdb553867ae7ce7be",{}},[name](Runtime& r){return std::make_unique<ComponentOperators>(r,name);});}
}
