#include "cipheropbench/component_plans.hpp"
#include <iostream>
#include <random>
using namespace cipheropbench::component;
void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
int main(){
 auto pp=polynomial_plan(128,768,768,32768);check(pp.batch==128&&pp.input==16&&pp.output==16,"Private-LLM block selector");auto ip=imra_plan(256,256,1024);check(ip.baby==8&&ip.horizontal==2&&ip.giants()==16&&ip.rotations==25,"Published IMRA search");
 for(int n:{32,128,512}){std::vector<Complex> a(n),b(n),expected(n);for(int i=0;i<n;++i){a[i]=std::sin(i*.7)*.1;b[i]=std::cos(i*.3)*.07;}
  for(int i=0;i<n;++i)for(int j=0;j<n;++j)expected[(i+j)%n]+=(i+j>=n?-1.:1.)*a[i]*b[j];
  auto af=apply_fourier(a,fourier_stages(n,false)),bf=apply_fourier(b,fourier_stages(n,false));for(int i=0;i<n;++i)af[i]*=bf[i]*std::sqrt(double(n));auto result=apply_fourier(af,fourier_stages(n,true));double error=0;for(int i=0;i<n;++i)error=std::max(error,std::abs(result[i]-expected[i]));check(error<1e-11,"Fused negacyclic Fourier convolution");
 }
 // Run horizontal/vertical IMRA literally, against an independent dense product.
 for(int n:{8,32,256}){int capacity=n*4;auto p=imra_plan(n,n,capacity);int gs=p.giants(),sf=gs*p.baby;std::vector<double> x(n),w(n*n),z(capacity),expected(n);
  for(int i=0;i<n;++i){x[i]=std::sin(i+.2);for(int j=0;j<n;++j)w[i*n+j]=std::cos(i*3+j*.2)*.03;}
  for(int j=0;j<n;++j)for(int i=0;i<n;++i)expected[j]+=x[i]*w[i*n+j];
  for(int g=0;g<gs;++g)for(int b=0;b<p.baby;++b)for(int j=0;j<capacity;++j){int src=(j+g*p.baby)%capacity,k=src/(n+sf),local=src%(n+sf);if(k>=p.horizontal)continue;int in=(local+k*sf+b)%n,out=(local-g*p.baby+n)%n;z[j]+=x[(src+b)%n]*w[in*n+out];}
  for(int j=0;j<n;++j){double y=0;for(int k=0;k<p.horizontal;++k)y+=z[j+k*(n+sf)];check(std::abs(y-expected[j])<1e-11,"IMRA cyclic embeddings and reduction");}
 }
 std::cout<<"Polynomial convolution, block search and horizontal/vertical IMRA PASS\n";
}
