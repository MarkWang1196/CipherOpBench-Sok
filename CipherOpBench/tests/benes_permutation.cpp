#include "cipheropbench/benes_permutation.hpp"
#include <complex>
#include <iostream>
#include <numeric>
#include <random>

void verify(const std::vector<int>& p){
  auto layers=cipheropbench::packing::permutation_network(p);
  std::vector<std::complex<double>> x(p.size());
  for(std::size_t i=0;i<x.size();++i)x[i]={double(i),double(2*i+1)};
  auto original=x;
  for(const auto& layer:layers){
    auto next=x;
    for(std::size_t i=0;i<x.size();++i)if(layer.swapped[i])next[i]=x[i^layer.distance];
    x=std::move(next);
  }
  for(std::size_t i=0;i<x.size();++i)if(x[i]!=original[p[i]])throw std::runtime_error("Permutation output mismatch");
}
int main(){
  std::mt19937 rng(20261106);std::size_t checked=0;
  for(int bits=0;bits<=15;++bits){
    std::vector<int> p(std::size_t(1)<<bits);std::iota(p.begin(),p.end(),0);verify(p);
    for(int attempt=0;attempt<3;++attempt){std::shuffle(p.begin(),p.end(),rng);verify(p);checked+=p.size();}
  }
  // The partial two-head canonical-to-CRT query map is completed with unused lanes.
  std::vector<int> p(32768,-1);std::vector<bool> used(p.size());
  for(int i=0;i<2*65*67;++i){int h=i%2,t=i%65,c=i%67;if(t<64&&c<64){p[i]=h*16384+t*128+c;used[p[i]]=true;}}
  int unused=0;for(auto& i:p)if(i<0){while(used[unused])++unused;i=unused;used[unused]=true;}
  verify(p);checked+=p.size();
  for(auto bad:std::vector<std::vector<int>>{{},{0,1,2},{0,0},{0,2},{-1,0}}){
    bool rejected=false;try{cipheropbench::packing::permutation_network(bad);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("Invalid mapping accepted");
  }
  std::cout<<"Passed complete complex-slot permutation checks: "<<checked<<std::endl;
}
