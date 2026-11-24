#pragma once
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace cipheropbench::calibration {
struct Layer {
 double gelu_bound=0,softmax_low=0,softmax_shift=0;
 int exp_squarings=0,euston_softmax_squarings=0,euston_gelu_squarings=0;
 std::vector<double> gelu_coefficients;
 std::array<double,2> low{},high{};
 std::array<int,2> iterations{};
};
struct Profile {
 bool enabled=false;
 std::string method,task,model,revision,path;
 double epsilon=0;
 std::array<Layer,12> layers;
};
inline Profile load(const std::string& path,const std::string& method){
 Profile p;if(path.empty())return p;
 std::ifstream in(path);std::string magic;int count;
 if(!(in>>magic>>p.method>>p.task>>p.model>>p.revision>>p.epsilon>>count)||magic!="CIPHEROPBENCH_PUBLIC_CALIBRATION_V1"||p.method!=method||count!=12||p.epsilon<0||!std::isfinite(p.epsilon))throw std::runtime_error("Invalid calibration header or method");
 for(int i=0;i<12;++i){auto& l=p.layers[i];int index,size;
  if(!(in>>index>>l.gelu_bound>>size)||index!=i||l.gelu_bound<10||l.gelu_bound>10000||!std::isfinite(l.gelu_bound)||size<2||size>512)throw std::runtime_error("Invalid GELU calibration");
  l.gelu_coefficients.resize(size);for(auto& c:l.gelu_coefficients)if(!(in>>c)||!std::isfinite(c))throw std::runtime_error("Invalid polynomial coefficient");
  if(!(in>>l.softmax_low>>l.softmax_shift>>l.exp_squarings>>l.euston_softmax_squarings>>l.euston_gelu_squarings)||!std::isfinite(l.softmax_low)||!std::isfinite(l.softmax_shift)||l.softmax_low>=l.softmax_shift||l.exp_squarings<7||l.exp_squarings>16||l.euston_softmax_squarings<1||l.euston_softmax_squarings>20||l.euston_gelu_squarings<1||l.euston_gelu_squarings>20)throw std::runtime_error("Invalid exponential calibration");
  for(int k=0;k<2;++k)if(!(in>>l.low[k]>>l.high[k]>>l.iterations[k])||!std::isfinite(l.low[k])||!std::isfinite(l.high[k])||l.low[k]<=0||l.high[k]<=l.low[k]||l.iterations[k]<1||l.iterations[k]>40)throw std::runtime_error("Invalid LayerNorm calibration");
 }
 std::string extra;if(in>>extra)throw std::runtime_error("Unexpected calibration data");
 p.enabled=true;p.path=path;return p;
}
inline std::vector<double> read_reference(const std::string& path,size_t count){
 std::ifstream in(path,std::ios::binary);std::vector<double> values(count);
 if(!in.read(reinterpret_cast<char*>(values.data()),count*sizeof(double))||in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Approximate reference byte count mismatch");
 for(double x:values)if(!std::isfinite(x))throw std::runtime_error("Non-finite approximate reference");
 return values;
}

}
