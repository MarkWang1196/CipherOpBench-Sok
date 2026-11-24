#pragma once
#include "research_math.cuh"
#include "cipheropbench/benes_permutation.hpp"
#include <numeric>
#include <climits>
#include <set>
namespace cipheropbench::detail {
// Dense injective CRT gathers use a permutation network; sparse/replicated maps retain BSGS.
inline Cipher tricycle_route(Math& m,const CipherList& input,const std::function<std::tuple<int,int,double>(int)>& mapping){
 std::map<std::pair<int,int>,std::vector<std::pair<int,double>>> groups;
 for(int i=0;i<slots;++i){auto [source,index,value]=mapping(i);if(source>=0&&value!=0)groups[{source,(index-i+slots)%slots}].emplace_back(i,value);}
 if(groups.size()<64)return m.route(input,mapping);
 if(input.size()==1){
  std::vector<int> permutation(slots,-1);std::vector<uint8_t> used(slots);Vec factors(slots);bool injective=true;
  for(auto& [key,positions]:groups)for(auto [position,value]:positions){
   int index=(position+key.second)%slots;
   if(key.first!=0||used[index]){injective=false;break;}
   permutation[position]=index;used[index]=1;factors[position]=value;
  }
  if(injective){
   int next=0;for(auto& index:permutation)if(index<0){while(used[next])++next;index=next;used[next]=1;}
   auto network=packing::permutation_network(permutation);auto value=input[0];Math layout{m.r};layout.prime_scale=true;
   for(const auto& layer:network){
    if(std::none_of(layer.swapped.begin(),layer.swapped.end(),[](uint8_t x){return x!=0;}))continue;
    value=layout.route({value},[&](int i){return std::tuple{0,layer.swapped[i]?int(i^layer.distance):i,1.};});
   }
   return layout.pm(std::move(value),factors);
  }
 }
 int width=16,best=INT_MAX;
 for(int candidate:{16,32,64,128,256,512}){
  std::set<std::pair<int,int>> babies;std::set<int> giants;
  for(auto& [key,positions]:groups){if(key.second%candidate)babies.emplace(key.first,key.second%candidate);if(key.second/candidate)giants.insert(key.second/candidate);}
  int cost=int(babies.size()+giants.size());if(cost<best){best=cost;width=candidate;}
 }
 CipherList prepared=input;for(auto& c:prepared)m.need(c);
 std::map<std::pair<int,int>,Cipher> babies;
 std::map<int,std::vector<std::pair<int,int>>> giants;
 for(auto& [key,positions]:groups){auto baby=std::pair{key.first,key.second%width};if(!babies.count(baby))babies.emplace(baby,m.rot(prepared.at(baby.first),baby.second));giants[key.second/width].push_back(key);}
 Cipher result;bool initial=true;
 for(auto& [giant,keys]:giants){Cipher bucket;bool first=true;int shift=giant*width;
  for(auto key:keys){Vec mask(slots);for(auto [position,value]:groups.at(key))mask[(position+shift)%slots]=value;
   auto term=m.pm(babies.at({key.first,key.second%width}),mask);bucket=first?std::move(term):m.add(bucket,term);first=false;
  }
  auto term=m.rot(bucket,shift);result=initial?std::move(term):m.add(result,term);initial=false;
 }
 return result;
}
inline int tricycle_inverse(int a,int n){for(int i=1;i<n;++i)if((a*i)%n==1)return i;TORCH_CHECK(false,"Tricycle dimensions must be pairwise coprime");}
inline int tricycle_index(int h,int row,int col,int H,int M,int N){
 TORCH_CHECK(std::gcd(H,M)==1&&std::gcd(H,N)==1&&std::gcd(M,N)==1,"Non-bijective CRT dimensions");
 for(int i=col;i<H*M*N;i+=N)if(i%H==h&&i%M==row)return i;TORCH_CHECK(false,"CRT index missing");
}
inline Cipher tricycle_shift(Math& m,const Cipher& x,int period,int amount,int length){
 amount=(amount%period+period)%period;TORCH_CHECK(period>0&&length<=slots,"CRT period exceeds slots");
 return tricycle_route(m,{x},[=](int i){if(i>=length)return std::tuple{-1,0,0.};return std::tuple{0,(i+amount)%period,1.};});
}
inline CipherList tricycle_linear(Math& m,CipherList input,const LinearWeights& w){
 for(auto& c:input)m.need(c,10);constexpr int M=129,N=128,L=M*N,B=16;auto weight=w.weight.contiguous();auto a=weight.accessor<double,2>();
 std::vector<CipherList> baby(6);
 for(int k=0;k<6;++k){auto encoded=tricycle_route(m,{input[k/2]},[=](int i){int t=i%M,c=i%N;if(i>=L||t>=128)return std::tuple{-1,0,0.};return std::tuple{0,(k%2)*half+t*128+c,1.};});
  m.need(encoded,4);
  for(int b=0;b<B;++b)baby[k].push_back(tricycle_shift(m,encoded,L,b*M,L));}
 CipherList result(3);bool first[3]{true,true,true};
 for(int out=0;out<6;++out){Cipher total;bool initial=true;
  for(int g=0;g<8;++g){Cipher bucket;bool empty=true;
   for(int k=0;k<6;++k)for(int b=0;b<B;++b){Vec c(slots);bool nonzero=false;int shift=g*B*M;
    for(int i=0;i<L;++i){int j=(i-shift%L+L)%L;if(j%M>=128)continue;double v=a[k*128+(j+g*B+b)%N][out*128+j%N];c[i]=v;nonzero|=v!=0.;}
    if(!nonzero)continue;auto pt=k::CKKSEncode(c,m.r.ckks,&baby[k][b],true);Cipher term;m.r.ckks->evaluator.multiply_plain(baby[k][b],pt,term);
    if(empty)bucket=std::move(term);else m.r.ckks->evaluator.add_inplace(bucket,term);empty=false;
   }if(empty)continue;m.r.ckks->evaluator.rescale_to_next_inplace(bucket);bucket.set_scale(k::scale);auto term=tricycle_shift(m,bucket,L,g*B*M,L);total=initial?std::move(term):m.add(total,term);initial=false;
  }if(initial)total=m.zero(input[0]);auto decoded=tricycle_route(m,{total},[=](int i){int h=i/half,t=(i%half)/128,c=i%128;if(h!=out%2)return std::tuple{-1,0,0.};return std::tuple{0,tricycle_index(0,t,c,1,M,N),1.};});
  int group=out/2;result[group]=first[group]?std::move(decoded):m.add(result[group],decoded);first[group]=false;
 }add_bias(m,result,w.bias);return result;
}
// Native right/left BSGS with replicated CRT periods and giant-shifted masks.
inline Cipher tricycle_cc_tile(Math& m,Cipher a,Cipher b,int H,int M,int N,int P,bool left_bsgs=false){
 m.need(a,5);m.need(b,5);
 int alen=H*M*N,blen=H*N*P,outlen=H*M*P;
 int ra=1+(outlen-H*M+alen-1)/alen,rb=1+(outlen-H*P+blen-1)/blen;
 TORCH_CHECK(ra*alen<=slots&&rb*blen<=slots&&outlen<=slots,"Insufficient common slots for Tricycle CRT repetitions");
 a=tricycle_route(m,{a},[=](int i){if(i>=ra*alen)return std::tuple{-1,0,0.};return std::tuple{0,i%alen,1.};});
 b=tricycle_route(m,{b},[=](int i){if(i>=rb*blen)return std::tuple{-1,0,0.};return std::tuple{0,i%blen,1.};});
 int sa=H*M*tricycle_inverse(H*M,N),sb=H*P*tricycle_inverse(H*P,N),stride=left_bsgs?H*M:H*P;
 int baby=1,cost=INT_MAX,wb=left_bsgs?2:1,wg=left_bsgs?1:2;
 for(int x=1;x<=N;++x){int current=wb*x+wg*((N+x-1)/x);if(current<=cost){cost=current;baby=x;}}
 CipherList babies;for(int j=0;j<baby;++j)babies.push_back(m.rot(left_bsgs?a:b,j*stride));Cipher result;bool first=true;
 for(int start=0;start<N;start+=baby){int giant=start*stride;Vec mask(slots);for(int i=0;i<outlen;++i)mask[(i+giant)%slots]=1.;Cipher bucket;bool initial=true;
  for(int j=0;j<baby&&start+j<N;++j){int c=((start+j)*stride)%N;
   int other=(left_bsgs?sb*c%blen:sa*c%alen)-giant;
   auto changed=m.pm(m.rot(left_bsgs?b:a,other),mask),cached=babies[j];m.align(changed,cached);Cipher term;
   m.r.ckks->evaluator.multiply(changed,cached,term);
   if(initial)bucket=std::move(term);else m.r.ckks->evaluator.add_inplace(bucket,term);initial=false;
  }
  m.r.ckks->evaluator.relinearize_inplace(bucket,*m.r.ckks->relin_keys);m.r.ckks->evaluator.rescale_to_next_inplace(bucket);
  if(m.prime_scale)bucket=m.pm(std::move(bucket),Vec(slots,1.));else bucket.set_scale(k::scale);
  auto term=m.rot(bucket,giant);result=first?std::move(term):m.add(result,term);first=false;
 }return result;
}
inline Cipher tricycle_qk(Math& m,Cipher q,Cipher key){
 m.need(q,9);m.need(key,9);Cipher scores;bool first=true;
 for(int row=0;row<2;++row)for(int col=0;col<2;++col){
  auto a=tricycle_route(m,{q},[=](int i){int h=i%2,t=i%65,c=i%67;if(i>=2*65*67||t>=64||c>=64)return std::tuple{-1,0,0.};return std::tuple{0,h*half+(row*64+t)*128+c,1.};});
  auto b=tricycle_route(m,{key},[=](int i){int h=i%2,c=i%67,t=i%69;if(i>=2*67*69||t>=64||c>=64)return std::tuple{-1,0,0.};return std::tuple{0,h*half+(col*64+t)*128+c,1.};});
  auto z=tricycle_cc_tile(m,a,b,2,65,67,69);
  auto decoded=tricycle_route(m,{z},[=](int i){int h=i/half,t=(i%half)/128,c=i%128;if(t/64!=row||c/64!=col)return std::tuple{-1,0,0.};return std::tuple{0,tricycle_index(h,t%64,c%64,2,65,69),.125};});scores=first?std::move(decoded):m.add(scores,decoded);first=false;
 }return m.row_diagonal(scores,false);
}
inline Cipher tricycle_av(Math& m,Cipher p,Cipher v){
 m.integer_scale(p,32);m.need(p,11);m.need(v,9);auto rows=m.row_diagonal(p,true);Cipher output;bool first=true;
 for(int row=0;row<2;++row){Cipher total;
  for(int k=0;k<2;++k){auto a=tricycle_route(m,{rows},[=](int i){int h=i%2,t=i%65,c=i%67;if(i>=2*65*67||t>=64||c>=64)return std::tuple{-1,0,0.};return std::tuple{0,h*half+(row*64+t)*128+k*64+c,1.};});
   auto b=tricycle_route(m,{v},[=](int i){int h=i%2,t=i%67,c=i%69;if(i>=2*67*69||t>=64||c>=64)return std::tuple{-1,0,0.};return std::tuple{0,h*half+(k*64+t)*128+h*64+c,1.};});auto term=tricycle_cc_tile(m,a,b,2,65,67,69,true);total=k?m.add(total,term):std::move(term);
  }auto decoded=tricycle_route(m,{total},[=](int i){int h=i/half,t=(i%half)/128,c=i%128;if(t/64!=row||c/64!=h)return std::tuple{-1,0,0.};return std::tuple{0,tricycle_index(h,t%64,c%64,2,65,69),1./32};});output=first?std::move(decoded):m.add(output,decoded);first=false;
 }return output;
}
}
