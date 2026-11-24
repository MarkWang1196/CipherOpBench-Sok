#pragma once
#include "cipheropbench/calibrated_math.hpp"
struct ScalarMath {
 void need(double&,int=1){}
 double add(double a,double b){return a+b;}double sub(double a,double b){return a-b;}
 double addc(double a,double b){return a+b;}double scale(double a,double b){return a*b;}
 double mul(double a,double b){return a*b;}double square(double a){return a*a;}double zero(double){return 0.;}
 double poly(double x,const std::vector<double>& c){double y=0;for(auto i=c.rbegin();i!=c.rend();++i)y=y*x+*i;return y;}
 double inverse(double x,double divisor,int count){double a=1-x/divisor,y=1+a;for(int i=0;i<count;++i){a=a*a;y*=1+a;}return y/divisor;}
};
