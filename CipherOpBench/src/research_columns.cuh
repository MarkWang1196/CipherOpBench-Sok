#pragma once
#include "research_math.cuh"
namespace cipheropbench::detail {
inline CipherList research_column_linear(Math& m,CipherList x,const LinearWeights& w,bool dual){
 auto input=spatial_inputs(m,std::move(x));auto weights=w.weight.contiguous();auto a=weights.accessor<double,2>();int groups=dual?2:3;CipherList output(groups);std::vector<bool> initialized(groups,false);
 for(int f=0;f<768;++f){bool used=false;for(int c=0;c<768;++c)used|=a[f][c]!=0.;if(!used)continue;auto col=column(m,input,f);
  for(int g=0;g<groups;++g){k::FlatVec coefficients(slots);bool nonzero=false;
   for(int i=0;i<slots;++i){int lane=i/128;if(dual){int feature=g*512+(lane/64)*128+lane%64;double re=feature<768?a[f][feature]:0.,im=feature+64<768?a[f][feature+64]:0.;coefficients[i]=make_cuDoubleComplex(re,im);nonzero|=re!=0||im!=0;}else{double v=a[f][g*256+lane];coefficients[i]=make_cuDoubleComplex(v,0.);nonzero|=v!=0.;}}
   if(!nonzero)continue;auto pt=k::CKKSEncode(coefficients,m.r.ckks,&col,true);Cipher term;m.r.ckks->evaluator.multiply_plain(col,pt,term);
   if(!initialized[g])output[g]=std::move(term);else m.r.ckks->evaluator.add_inplace(output[g],term);initialized[g]=true;
  }
 }
 for(int g=0;g<groups;++g){if(initialized[g]){m.r.ckks->evaluator.rescale_to_next_inplace(output[g]);output[g].set_scale(k::scale);}else output[g]=m.zero(input[0]);}
 if(dual){CipherList components;for(auto& c:output){components.push_back(real_part(m,c));components.push_back(imaginary_part(m,c));}output.clear();
  for(int g=0;g<3;++g)output.push_back(m.route(components,[=](int i){int f=g*256+(i/half)*128+i%128,t=(i%half)/128,lane=((f%512)/128)*64+f%64;return std::tuple{(f/512)*2+(f/64)%2,lane*128+t,1.};}));
 }else for(auto& c:output)c=shallow_tile_transpose(m,c);
 add_bias(m,output,w.bias);return output;
}
inline double ternary_scale(const LinearWeights& w){auto a=w.weight;double scale=a.abs().max().item<double>();if(scale==0.)return 1.;return ((a==0)|(a==scale)|(a==-scale)).all().item<bool>()?scale:0.;}
inline CipherList ensi_bitlinear(Math& m,CipherList x,const LinearWeights& w){
 auto spatial=spatial_inputs(m,std::move(x));auto weight=w.weight.contiguous();auto a=weight.accessor<double,2>();CipherList out;
 // Stream one output column: the native add/negate path never quantizes shared weights.
 for(int g=0;g<3;++g){Cipher sum;bool group_first=true;
  for(int j=0;j<256;++j){Cipher col;bool first=true;
   for(int f=0;f<768;++f){double v=a[f][g*256+j];if(v==0)continue;auto term=column(m,spatial,f);if(v<0)m.r.ckks->evaluator.negate_inplace(term);col=first?std::move(term):m.add(col,term);first=false;}
   if(first)col=m.zero(spatial[0]);Vec mask(slots);std::fill_n(mask.begin()+j*128,128,1.);auto term=m.pm(col,mask);sum=group_first?std::move(term):m.add(sum,term);group_first=false;
  }out.push_back(tile_transpose(m,sum));
 }double scale=ternary_scale(w);if(scale!=1.)for(auto& c:out)c=m.scale(c,scale);add_bias(m,out,w.bias);return out;
}
inline Cipher stip_qk(Math& m,Cipher q,Cipher key){
 m.need(q,8);m.need(key,8);auto hybrid=m.add(key,thor_imaginary(m,head_row_rotation(m,key,64)));Cipher result;
 for(int d=0;d<64;++d){auto products=m.mul(q,head_row_rotation(m,hybrid,d));auto sum=m.row_sum(products);auto re=real_part(m,sum),im=imaginary_part(m,sum);Vec lo(slots),hi(slots);
  for(int i=0;i<slots;++i){lo[i]=i%128==d?1./8:0.;hi[i]=i%128==d+64?1./8:0.;}auto term=m.add(m.pm(re,lo),m.pm(im,hi));result=d?m.add(result,term):std::move(term);
 }return result;
}
inline Cipher stip_av(Math& m,Cipher p,Cipher v){
 m.integer_scale(p,128);m.need(p,9);m.need(v,9);
 auto real=m.route({v},[](int i){int h=i/half,t=(i%half)/128,c=i%128;if(c>=32)return std::tuple{-1,0,0.};return std::tuple{0,h*half+t*128+h*64+c,1.};});
 auto imag=m.route({v},[](int i){int h=i/half,t=(i%half)/128,c=i%128;if(c>=32)return std::tuple{-1,0,0.};return std::tuple{0,h*half+t*128+h*64+c+32,1.};});
 auto z=m.add(real,thor_imaginary(m,imag));CipherList babies;for(int b=0;b<8;++b)babies.push_back(head_row_rotation(m,z,b));Cipher total;
 for(int g=0;g<16;++g){Cipher bucket;
  for(int b=0;b<8;++b){int d=g*8+b;auto prob=m.route({p},[=](int i){int h=i/half,t=(i%half)/128;return std::tuple{0,h*half+((t-8*g+128)%128)*128+d,1.};});auto term=m.mul(prob,babies[b]);bucket=b?m.add(bucket,term):std::move(term);}
  bucket=head_row_rotation(m,bucket,8*g);total=g?m.add(total,bucket):std::move(bucket);
 }
 auto re=real_part(m,total),im=imaginary_part(m,total);
 return m.route({re,im},[](int i){int h=i/half,t=(i%half)/128,c=i%128;if(c/64!=h)return std::tuple{-1,0,0.};return std::tuple{(c%64)/32,h*half+t*128+c%32,1./128};});
}
inline Cipher ensi_qk(Math& m,Cipher q,Cipher key){
 m.need(q,8);m.need(key,8);Cipher rows;
 for(int j=0;j<128;++j){auto repeated=repeated_row(m,key,j);auto product=m.mul(q,repeated);auto sum=m.row_sum(product);Vec at(slots);for(int i=0;i<slots;++i)at[i]=i%128==j?.125:0.;auto term=m.pm(sum,at);rows=j?m.add(rows,term):std::move(term);}
 return m.row_diagonal(rows,false);
}
inline Cipher ensi_av(Math& m,Cipher p,Cipher v){
 m.integer_scale(p,128);m.integer_scale(v,4);m.need(p,12);m.need(v,12);p=m.row_diagonal(p,true);
 auto input=m.route({v},[](int i){int h=i/half,t=(i%half)/128,c=i%128;if(c>=64)return std::tuple{-1,0,0.};return std::tuple{0,h*half+t*128+h*64+c,1.};});input=tile_transpose(m,input);Cipher output;
 for(int c=0;c<64;++c){auto repeated=repeated_row(m,input,c);auto sum=m.row_sum(m.mul(p,repeated));Vec mask(slots);for(int i=0;i<slots;++i)mask[i]=i%128==(i/half)*64+c?1./512:0.;auto term=m.pm(sum,mask);output=c?m.add(output,term):std::move(term);}
 return output;
}
}
