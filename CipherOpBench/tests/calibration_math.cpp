#include "calibration_scalar.hpp"
#include <iostream>
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("Expected profile path");
 auto profile=cipheropbench::calibration::load(argv[1],"nexus");ScalarMath m;auto p=profile.layers[0];
 for(auto method:{"nexus","moai","euston","thor"}){
  double maximum=0.;for(int i=0;i<=36000;++i){double x=-180+i*.01;
   double y=cipheropbench::calibration::gelu(m,x,p,method),exact=.5*x*(1+std::erf(x/std::sqrt(2.)));
   if(!std::isfinite(y))throw std::runtime_error("Non-finite in-domain GELU");maximum=std::max(maximum,std::abs(y-exact));}
  std::cout<<method<<" GELU oracle max error="<<maximum<<std::endl;
  if(maximum>(std::string(method)=="euston"?.04:.01))throw std::runtime_error("GELU oracle mismatch");
 }
 for(auto method:{"nexus","moai","euston","thor"})for(double v:{.03,.3,1.,10.,1000.}){
  p.low[0]=.005;p.high[0]=2000.;p.iterations[0]=24;
  auto y=cipheropbench::calibration::inverse_sqrt(m,v,p,1,method);
  if(!std::isfinite(y)||std::abs(y*std::sqrt(v)-1)>.002)throw std::runtime_error("Inverse sqrt oracle mismatch");
 }
 bool rejected=false;try{cipheropbench::calibration::load(argv[1],"moai");}catch(const std::exception&){rejected=true;}
 if(!rejected)throw std::runtime_error("Cross-method calibration accepted");
 std::cout<<"Calibration parser and mathematical oracles passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
