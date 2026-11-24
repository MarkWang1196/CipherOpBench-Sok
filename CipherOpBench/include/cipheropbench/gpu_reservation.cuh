#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <dlfcn.h>

namespace cipheropbench {
inline void reserve_gpu_memory() {
  const char* setting=std::getenv("CIPHEROPBENCH_GPU_RESERVE_MIB");
  if(!setting||!*setting)return;
  std::string text(setting);size_t parsed=0;uint64_t mib=std::stoull(text,&parsed);
  if(parsed!=text.size()||text[0]=='-'||mib>1048576)throw std::invalid_argument("Invalid CIPHEROPBENCH_GPU_RESERVE_MIB");
  if(!mib)return;
  if(std::getenv("CIPHEROPBENCH_GPU_ARENA")){
    auto query=reinterpret_cast<uint64_t(*)()>(dlsym(RTLD_DEFAULT,"cipheropbench_ipc_reserved_bytes"));
    if(!query||query()<mib*1048576)throw std::runtime_error("Persistent IPC owner has insufficient memory or allocator is not loaded");
    std::cout<<"GPU reservation: "<<query()/1048576<<" MiB held by independent persistent CUDA IPC owner"<<std::endl;return;
  }
  auto check=[](cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));};
  int device;check(cudaGetDevice(&device));cudaMemPool_t pool;check(cudaDeviceGetDefaultMemPool(&pool,device));
  uint64_t threshold=UINT64_MAX;check(cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&threshold));
  uint64_t reserved=0;check(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
  const uint64_t target=mib*1024*1024;
  if(reserved<target){
    void* buffer=nullptr;
    for(;;){
      auto code=cudaMallocAsync(&buffer,target,nullptr);
      if(code==cudaSuccess)break;
      if(code!=cudaErrorMemoryAllocation)check(code);
      cudaGetLastError();
      std::cerr<<"WAIT: cannot reserve "<<mib<<" MiB; retry in 30 seconds"<<std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(30));
    }
    check(cudaFreeAsync(buffer,nullptr));check(cudaDeviceSynchronize());
  }
  check(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
  if(reserved<target)throw std::runtime_error("CUDA pool did not retain requested reservation");
  std::cout<<"GPU reservation: "<<reserved/1048576<<" MiB retained in reusable CUDA async pool"<<std::endl;
}
}
