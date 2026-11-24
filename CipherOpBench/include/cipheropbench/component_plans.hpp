#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
namespace cipheropbench::component {
struct PolynomialPlan {int batch=0,input=0,output=0,degree=0;};
inline int ceil_div(int a,int b){return (a+b-1)/b;}
// Private-LLM seal-cuda/app/LinearHelper.cuh: minimize transmitted input/output blocks.
inline PolynomialPlan polynomial_plan(int batch,int input,int output,int capacity){
 PolynomialPlan p;int best=std::numeric_limits<int>::max();
 for(int b=batch;b>=1;--b){int bc=ceil_div(batch,b);if(b>=capacity||bc*2>best)continue;
  for(int i=1;i<capacity/b;++i){int o=std::min(capacity/b/i,output);if(i>input||o<1)continue;int cost=bc*(ceil_div(input,i)+ceil_div(output,o));
   if(cost<best){best=cost;p={b,i,o,capacity};}
  }
 }if(!p.batch)throw std::invalid_argument("No polynomial packing fits");return p;
}
struct ImraPlan {int baby=0,superblocks=0,horizontal=0,rotations=0;int giants()const{return superblocks/horizontal;}};
// MetaKernel vector_utils.cxx::Imra_cost_model, including its horizontal-batch tie break.
inline ImraPlan imra_plan(int nd,int kd,int capacity){
 ImraPlan best;best.rotations=nd*kd;
 for(int b=1;b<=nd;++b)if(nd%b==0){int pb=nd/b;for(int ps=1;ps<=pb;++ps){
  if(pb%ps||(kd*ps+nd>capacity&&kd!=capacity)||(kd==capacity&&ps>1))continue;
  int rep=ceil_div(kd*ps+nd,kd),cost=int(std::ceil(std::log2(rep)))+b-1+pb/ps-1+ps-1;
  if(cost<best.rotations||(cost==best.rotations&&ps>best.horizontal))best={b,pb,ps,cost};
 }}if(!best.baby)throw std::invalid_argument("No IMRA packing fits");return best;
}
using Complex=std::complex<double>;
using FourierStage=std::map<int,std::vector<Complex>>;
// Three fused butterflies per level; DIF forward and DIT inverse need no bit reversal.
inline std::vector<FourierStage> fourier_stages(int n,bool inverse){
 if(n<2||(n&(n-1)))throw std::invalid_argument("Fourier degree must be a power of two");
 std::vector<int> lengths;for(int l=inverse?2:n;inverse?l<=n:l>=2;l=inverse?l*2:l/2)lengths.push_back(l);
 std::vector<FourierStage> result;
 for(int begin=0;begin<int(lengths.size());begin+=3){int end=std::min(begin+3,int(lengths.size()));FourierStage diagonals;
  for(int j=0;j<n;++j){std::map<int,Complex> row{{j,1.}};
   for(int stage=end-1;stage>=begin;--stage){int length=lengths[stage],h=length/2;std::map<int,Complex> next;
    for(auto [at,c]:row){int local=at%length,base=at-local,k=local%h;Complex w=std::polar(1.,(inverse?2.:-2.)*std::acos(-1.)*k/length);double norm=1/std::sqrt(2.);
     if(inverse){next[base+k]+=c*norm;next[base+k+h]+=c*(local<h?w:-w)*norm;}
     else if(local<h){next[at]+=c*norm;next[at+h]+=c*norm;}
     else{next[at-h]+=c*w*norm;next[at]-=c*w*norm;}
    }row=std::move(next);
   }
   for(auto [source,c]:row){if(!inverse&&begin==0)c*=std::polar(1.,-std::acos(-1.)*source/n);if(inverse&&end==int(lengths.size()))c*=std::polar(1.,std::acos(-1.)*j/n);
    int shift=(source-j+n)%n;auto& d=diagonals[shift];if(d.empty())d.resize(n);d[j]=c;
   }
  }result.push_back(std::move(diagonals));
 }return result;
}
inline std::vector<Complex> apply_fourier(std::vector<Complex> x,const std::vector<FourierStage>& stages){
 for(const auto& stage:stages){std::vector<Complex> y(x.size());for(const auto& [shift,d]:stage)for(int j=0;j<int(x.size());++j)y[j]+=x[(j+shift)%x.size()]*d[j];x=std::move(y);}return x;
}
}
