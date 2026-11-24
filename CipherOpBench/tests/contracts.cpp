#include "cipheropbench/operators.cuh"
#include <iostream>
int main() {
  torch::NoGradGuard guard;
  try {
    cipheropbench::Shape{}.validate_nexus();
    bool rejected=false;
    try {cipheropbench::Shape{32,128,2,512}.validate_nexus();}catch(const std::exception&){rejected=true;}
    TORCH_CHECK(rejected,"incompatible shapes must be rejected");
    auto input=torch::arange(128*768,torch::kDouble).view({128,768});
    auto packed=cipheropbench::k::row_pack_128x768(input);
    std::vector<torch::Tensor> blocks;
    for(auto& values:packed) {
      TORCH_CHECK(values.size()==32768,"slot count mismatch");
      std::vector<double> real;for(auto v:values)real.push_back(v.x);
      auto t=cipheropbench::k::tensor_from_vector(real,{2,128,128});
      blocks.push_back(t[0]);blocks.push_back(t[1]);
    }
    TORCH_CHECK(torch::equal(torch::cat(blocks,1),input),"hidden layout round trip failed");
    auto mask=torch::ones({128,128},torch::kBool);mask.slice(1,126)=false;
    auto packed_mask=cipheropbench::TransformerOperators::pack_score_mask(mask);
    for(int row=0;row<128;++row) for(int col=0;col<128;++col) {
      double expected=(col+row)%128>=126?-100.:0.;
      TORCH_CHECK(packed_mask[row*128+col].x==expected,"diagonal mask mapping failed");
    }
    auto fixture=cipheropbench::synthetic_fixture(2,2026);
    TORCH_CHECK(!torch::equal(fixture.layers[0].q.weight,fixture.layers[1].q.weight),"encoder layers must have independent weights");
    std::cout<<"shape, packing, diagonal masking and layer independence passed\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
