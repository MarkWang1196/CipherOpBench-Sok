#pragma once
#include "literature_math.cuh"
namespace cipheropbench::detail {
// Half-local transpose using seven swaps of row/column index bits.
inline Cipher tile_transpose(Math& m,Cipher x){
  m.need(x,8);
  for(int bit=0;bit<7;++bit){int a=1<<bit,b=1<<(bit+7);Vec same(slots),lo(slots),hi(slots);
    for(int i=0;i<slots;++i){bool p=i&a,q=i&b;same[i]=p==q;lo[i]=!p&&q;hi[i]=p&&!q;}
    x=m.add(m.pm(x,same),m.add(m.rot(m.pm(x,lo),b-a),m.rot(m.pm(x,hi),a-b)));
  }return x;
}
// Group independent bit swaps into two mask stages to avoid refreshing tiny outputs.
inline Cipher shallow_tile_transpose(Math& m,Cipher x){
  m.need(x,3);for(int start:{0,4}){int count=start==0?4:3;
    x=m.route({x},[=](int i){int j=i;for(int bit=start;bit<start+count;++bit)if(bool(i&(1<<bit))!=bool(i&(1<<(bit+7))))j^=(1<<bit)|(1<<(bit+7));return std::tuple{0,j,1.};});
  }return x;
}
inline Cipher inner_rotation(Math& m,const Cipher& x,int amount){
  int shift=(amount%128+128)%128;if(!shift)return x;
  Vec left(slots),right(slots);for(int i=0;i<slots;++i){left[i]=i%128<128-shift;right[i]=!left[i];}
  return m.add(m.pm(m.rot(x,shift),left),m.pm(m.rot(x,shift-128),right));
}
inline Cipher interleave_heads(Math& m,const Cipher& x,bool inverse=false,double factor=1.){
  return m.route({x},[=](int i){int h,k,t;
    if(!inverse){t=i%128;h=(i/128)%2;k=i/256;return std::tuple{0,h*half+k*128+t,factor};}
    h=i/half;k=(i%half)/128;t=i%128;return std::tuple{0,(k*2+h)*128+t,factor};
  });
}
inline Cipher blb_ccmm(Math& m,Cipher a,Cipher b){
  // Spatial A[:,k], row-preprocessed B[k,:], two interleaved independent heads.
  m.need(a,12);m.need(b,12);
  a=interleave_heads(m,tile_transpose(m,std::move(a)));
  b=m.route({b},[](int i){int h=i/half,k=(i%half)/128,t=i%128;return std::tuple{0,h*half+k*128+(t+k)%128,1.};});
  b=interleave_heads(m,b);
  m.r.refresh(a); // Restore depth before BSGS rather than refreshing a small output.
  CipherList baby;for(int j=0;j<8;++j)baby.push_back(inner_rotation(m,b,j));
  Cipher result;bool first=true;
  for(int g=0;g<16;++g){auto ag=inner_rotation(m,a,-8*g);Cipher bucket;
    for(int j=0;j<8;++j){auto term=m.mul(ag,baby[j]);term=m.rot(term,-j*256);bucket=j?m.add(bucket,term):std::move(term);}
    bucket=inner_rotation(m,bucket,8*g);bucket=m.rot(bucket,-8*g*256);
    result=first?std::move(bucket):m.add(result,bucket);first=false;
  }
  return m.transpose(interleave_heads(m,result,true));
}
inline CipherList blb_linear(Math& m,CipherList x,const LinearWeights& w){
  for(auto& ct:x)m.need(ct,12);
  CipherList spatial;for(auto& ct:x)spatial.push_back(tile_transpose(m,ct));
  std::vector<CipherList> baby(3);for(int k=0;k<3;++k)for(int j=0;j<16;++j)baby[k].push_back(m.rot(spatial[k],j*128));
  auto weight=w.weight.contiguous();auto values=weight.accessor<double,2>();CipherList out;
  for(int group=0;group<3;++group){Cipher sum;bool first=true;
    for(int giant=0;giant<16;++giant){Cipher bucket;bool initial=true;
      for(int k=0;k<3;++k)for(int b=0;b<16;++b){int shift=16*giant+b;Vec coefficients(slots);bool nonzero=false;
        for(int lane=0;lane<256;++lane){int column=(lane-16*giant+256)%256;double value=values[k*256+(column+shift)%256][group*256+column];nonzero|=value!=0.;std::fill_n(coefficients.begin()+lane*128,128,value);}
        if(!nonzero)continue;auto ct=baby[k][b];auto pt=k::CKKSEncode(coefficients,m.r.ckks,&ct,true);Cipher term;m.r.ckks->evaluator.multiply_plain(ct,pt,term);
        if(initial){bucket=std::move(term);initial=false;}else m.r.ckks->evaluator.add_inplace(bucket,term);
      }
      if(initial)continue;
      m.r.ckks->evaluator.rescale_to_next_inplace(bucket);bucket.set_scale(k::scale);bucket=m.rot(bucket,16*giant*128);
      sum=first?std::move(bucket):m.add(sum,bucket);first=false;
    }
    if(first)sum=m.zero(spatial[0]);Vec bias(slots);for(int c=0;c<256;++c)std::fill_n(bias.begin()+c*128,128,w.bias[group*256+c].item<double>());
    auto pt=k::CKKSEncode(bias,m.r.ckks,&sum);m.r.ckks->evaluator.add_plain_inplace(sum,pt);out.push_back(tile_transpose(m,sum));
  }return out;
}
inline Cipher repeated_row(Math& m,const Cipher& x,int row){
  Vec mask(slots);for(int h=0;h<2;++h)std::fill_n(mask.begin()+h*half,128,1.);
  auto y=m.pm(m.rot(x,row*128),mask);for(int step=128;step<half;step*=2)y=m.add(y,m.rot(y,-step));return y;
}
inline CipherList tiny_linear(Math& m,CipherList x,const LinearWeights& w){
  for(auto& ct:x)m.need(ct,12);
  CipherList transposed;for(auto ct:x)transposed.push_back(tile_transpose(m,ct));
  CipherList output(3);std::array<bool,3> first{true,true,true};
  auto weights=w.weight.contiguous();auto wa=weights.accessor<double,2>();
  for(int pair=0;pair<64;++pair){CipherList expanded;
    for(int k=0;k<6;++k){auto z=m.route({transposed[k/2]},[=](int i){int h=i/half,f=(i%half)/128;if(i%128)return std::tuple{-1,0,0.};return std::tuple{0,(k%2)*half+f*128+2*pair+h,1.};});
      for(int step=1;step<128;step*=2)z=m.add(z,m.rot(z,-step));expanded.push_back(std::move(z));}
    for(int block=0;block<6;++block){Cipher sum;bool initial=true;
      for(int k=0;k<6;++k){Vec weight(slots);bool nonzero=false;for(int i=0;i<slots;++i){int input=(i%half)/128,out=i%128;double v=wa[k*128+input][block*128+out];weight[i]=v;nonzero|=v!=0.;}if(!nonzero)continue;
        auto term=m.pm(expanded[k],weight);sum=initial?std::move(term):m.add(sum,term);initial=false;}
      if(initial)sum=m.zero(expanded[0]);for(int step=128;step<half;step*=2)sum=m.add(sum,m.rot(sum,step));
      Vec bias(slots);for(int i=0;i<slots;++i)bias[i]=w.bias[block*128+i%128].item<double>();auto p=k::CKKSEncode(bias,m.r.ckks,&sum);m.r.ckks->evaluator.add_plain_inplace(sum,p);
      auto packed=m.route({sum},[=](int i){int h=i/half,t=(i%half)/128,c=i%128;if(h!=block%2||t/2!=pair)return std::tuple{-1,0,0.};return std::tuple{0,(t%2)*half+c,1.};});
      int group=block/2;output[group]=first[group]?std::move(packed):m.add(output[group],packed);first[group]=false;
    }
  }return output;
}
// VecMatRC: repeated rows x column-major weights, then canonical output packing.
inline CipherList tiny_rc_linear(Math& m,CipherList x,const LinearWeights& w){
  for(auto& ct:x)m.need(ct,12);CipherList output(3);std::array<bool,3> first{true,true,true};
  auto weights=w.weight.contiguous();auto wa=weights.accessor<double,2>();
  for(int pair=0;pair<64;++pair){CipherList repeated;
    for(int k=0;k<6;++k){auto z=m.route({x[k/2]},[=](int i){int h=i/half,c=i%128;if((i%half)/128)return std::tuple{-1,0,0.};return std::tuple{0,(k%2)*half+(2*pair+h)*128+c,1.};});for(int step=128;step<half;step*=2)z=m.add(z,m.rot(z,-step));repeated.push_back(std::move(z));}
    for(int block=0;block<6;++block){Cipher sum;bool initial=true;
      for(int k=0;k<6;++k){Vec coefficients(slots);bool nonzero=false;for(int i=0;i<slots;++i){int out=(i%half)/128,input=i%128;double value=wa[k*128+input][block*128+out];coefficients[i]=value;nonzero|=value!=0.;}if(!nonzero)continue;auto term=m.pm(repeated[k],coefficients);sum=initial?std::move(term):m.add(sum,term);initial=false;}
      if(initial)sum=m.zero(repeated[0]);for(int step=1;step<128;step*=2)sum=m.add(sum,m.rot(sum,step));
      Vec bias(slots);for(int i=0;i<slots;++i)bias[i]=w.bias[block*128+(i%half)/128].item<double>();auto p=k::CKKSEncode(bias,m.r.ckks,&sum);m.r.ckks->evaluator.add_plain_inplace(sum,p);
      Cipher packed;for(int h=0;h<2;++h){Vec mask(slots);for(int c=0;c<128;++c)mask[h*half+c*128]=1.;auto part=m.rot(m.pm(sum,mask),(h-block%2)*half-(2*pair+h));packed=h?m.add(packed,part):std::move(part);}
      int group=block/2;output[group]=first[group]?std::move(packed):m.add(output[group],packed);first[group]=false;
    }
  }for(auto& ct:output)ct=tile_transpose(m,ct);return output;
}
inline Cipher tiny_qk(Math& m,Cipher q,Cipher key){
  m.need(q,7);m.need(key,7);Cipher packed;
  Vec first_col(slots);for(int i=0;i<slots;++i)first_col[i]=i%128==0?1./8:0.;
  for(int row=0;row<128;++row){auto term=m.mul(repeated_row(m,q,row),key);for(int step=1;step<64;step*=2)term=m.add(term,m.rot(term,step));term=m.rot(m.pm(term,first_col),-row);packed=row?m.add(packed,term):std::move(term);}
  return m.row_diagonal(tile_transpose(m,packed),false);
}
inline Cipher tiny_av(Math& m,Cipher scores,Cipher value){
  m.need(scores,12);m.need(value,5);auto s=tile_transpose(m,m.row_diagonal(scores,true));Cipher output;
  Vec row_mask(slots);for(int i=0;i<slots;++i)row_mask[i]=(i%half)<128&&((i/half==0&&i%128<64)||(i/half==1&&i%128>=64));
  for(int row=0;row<128;++row){auto expanded=m.route({s},[=](int i){if(i%128)return std::tuple{-1,0,0.};return std::tuple{0,i+row,1.};});for(int step=1;step<128;step*=2)expanded=m.add(expanded,m.rot(expanded,-step));
    auto term=m.mul(expanded,value);for(int step=128;step<half;step*=2)term=m.add(term,m.rot(term,step));term=m.rot(m.pm(term,row_mask),-row*128);output=row?m.add(output,term):std::move(term);
  }return output;
}
}
