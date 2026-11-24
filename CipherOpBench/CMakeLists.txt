cmake_minimum_required(VERSION 3.26)
project(CipherOpBench VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
include(CTest)
if(BUILD_TESTING)
  foreach(test IN ITEMS calibration_math component_plans benes_permutation)
    add_executable(${test} tests/${test}.cpp)
    target_include_directories(${test} PRIVATE include)
  endforeach()
  add_test(NAME calibration_math COMMAND calibration_math
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/calibration.nexus.dat")
  add_test(NAME component_plans COMMAND component_plans)
  add_test(NAME benes_permutation COMMAND benes_permutation)
endif()

option(CIPHEROPBENCH_CPU_ONLY "Build the portable tests without CUDA" OFF)
if(CIPHEROPBENCH_CPU_ONLY)
  return()
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
option(CIPHEROPBENCH_UNIFIED_PHANTOM_GPU "Build the unified decoder profile" OFF)

find_package(CUDAToolkit REQUIRED)
find_package(PhantomBootstrapping 1.0 CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)
find_library(NTL_LIBRARY NAMES ntl REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)
set_property(TARGET PhantomBootstrapping::PhantomBootstrapping PROPERTY INTERFACE_LINK_LIBRARIES
  "${NTL_LIBRARY};${GMP_LIBRARY};m;pthread;PhantomBootstrapping::Phantom")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(cipheropbench_kernels SHARED
  src/kernels/matrix_mul.cu src/kernels/row_pack.cpp src/kernels/softmax.cu
  src/kernels/gelu.cu src/kernels/layer_norm.cu src/kernels/argmax.cu)
target_include_directories(cipheropbench_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(cipheropbench_kernels PUBLIC
  PhantomBootstrapping::PhantomBootstrapping ${TORCH_LIBRARIES} CUDA::cudart)

add_library(cipheropbench SHARED
  src/runtime.cu src/operators.cu src/literature.cu src/extended.cu src/research.cu
  src/column_methods.cu src/components.cu src/published.cu src/fixtures.cpp src/decoder.cu)
add_library(CipherOpBench::cipheropbench ALIAS cipheropbench)
set_source_files_properties(src/fixtures.cpp tests/contracts.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(cipheropbench PUBLIC cipheropbench_kernels PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(cipheropbench PRIVATE
  CIPHEROPBENCH_ATLAS_DEFAULT_PROFILE="${CMAKE_CURRENT_SOURCE_DIR}/configs/research/atlas-published-x.dat")

if(CIPHEROPBENCH_UNIFIED_PHANTOM_GPU)
  target_compile_definitions(cipheropbench_kernels PUBLIC CIPHEROPBENCH_UNIFIED_PHANTOM_GPU=1)
  add_library(cipheropbench_cachemir SHARED src/cachemir_gpu.cu)
  target_link_libraries(cipheropbench_cachemir PUBLIC cipheropbench_kernels)
  foreach(scope IN ITEMS components layer)
    add_executable(cipheropbench_cachemir_${scope} benchmarks/cachemir_gpu_${scope}.cu)
    target_include_directories(cipheropbench_cachemir_${scope} PRIVATE assets/cachemir_gpu)
    target_link_libraries(cipheropbench_cachemir_${scope} PRIVATE cipheropbench_cachemir)
  endforeach()
else()
  add_executable(cipheropbench_bench benchmarks/main.cu)
  add_executable(cipheropbench_bert_batch benchmarks/bert_arion_batch.cu)
  add_executable(cipheropbench_operator_batch benchmarks/bert_operator_batch.cu)
  add_executable(cipheropbench_decoder_bench benchmarks/decoder_main.cu)
  foreach(target IN ITEMS cipheropbench_bench cipheropbench_bert_batch
      cipheropbench_operator_batch cipheropbench_decoder_bench)
    target_link_libraries(${target} PRIVATE cipheropbench)
  endforeach()
endif()

if(BUILD_TESTING)
  add_executable(tensor_contract tests/contracts.cpp)
  target_link_libraries(tensor_contract PRIVATE cipheropbench)
  add_test(NAME tensor_contract COMMAND tensor_contract)
  if(CIPHEROPBENCH_UNIFIED_PHANTOM_GPU)
    foreach(test IN ITEMS unified_decoder_contracts unified_decoder_gqa_contracts)
      add_executable(${test} tests/${test}.cu)
      target_link_libraries(${test} PRIVATE cipheropbench)
    endforeach()
  endif()
endif()
