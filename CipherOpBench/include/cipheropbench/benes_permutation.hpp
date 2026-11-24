#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cipheropbench::packing {
struct PermutationLayer {
  std::size_t distance;
  std::vector<std::uint8_t> swapped;
};
namespace detail {
inline std::vector<PermutationLayer> benes(const std::vector<int>& p) {
  const auto n=p.size();
  if(n==1)return {};
  if(n==2)return {{1,std::vector<std::uint8_t>(2,p[0]!=0)}};
  std::vector<int> inverse(n),color(n,-1);
  for(std::size_t i=0;i<n;++i)inverse[p[i]]=int(i);
  // Alternating edge colors send one member of every input/output pair to each subnetwork.
  for(std::size_t seed=0;seed<n;++seed)if(color[seed]<0){
    color[seed]=0;std::vector<int> pending{int(seed)};
    while(!pending.empty()){
      auto edge=pending.back();pending.pop_back();
      for(auto mate:{edge^1,p[inverse[edge]^1]}){
        if(color[mate]<0){color[mate]=1-color[edge];pending.push_back(mate);}
        else if(color[mate]==color[edge])throw std::logic_error("Invalid permutation edge coloring");
      }
    }
  }
  PermutationLayer first{1,std::vector<std::uint8_t>(n)},last=first;
  std::vector<int> even(n/2),odd(n/2);
  for(std::size_t i=0;i<n/2;++i){
    first.swapped[2*i]=first.swapped[2*i+1]=color[2*i];
    last.swapped[2*i]=last.swapped[2*i+1]=color[p[2*i]];
    const auto at=2*i+color[p[2*i]];
    even[i]=p[at]/2;odd[i]=p[at^1]/2;
  }
  auto left=benes(even),right=benes(odd);
  std::vector<PermutationLayer> result;result.push_back(std::move(first));
  for(std::size_t j=0;j<left.size();++j){
    PermutationLayer layer{2*left[j].distance,std::vector<std::uint8_t>(n)};
    for(std::size_t i=0;i<n/2;++i){layer.swapped[2*i]=left[j].swapped[i];layer.swapped[2*i+1]=right[j].swapped[i];}
    result.push_back(std::move(layer));
  }
  result.push_back(std::move(last));return result;
}
}
// output[i] = input[p[i]]; each stage swaps disjoint pairs a power of two apart.
inline std::vector<PermutationLayer> permutation_network(const std::vector<int>& p){
  if(p.empty()||(p.size()&(p.size()-1)))throw std::invalid_argument("Permutation size must be a power of two");
  std::vector<std::uint8_t> used(p.size());
  for(auto i:p){if(i<0||std::size_t(i)>=p.size()||used[i])throw std::invalid_argument("Permutation must be bijective");used[i]=1;}
  return detail::benes(p);
}
}
