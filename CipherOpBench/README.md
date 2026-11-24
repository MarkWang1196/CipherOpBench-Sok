# CipherOpBench

CUDA benchmarks for encrypted Transformer operators. The source includes BERT-base
single-request and batch benchmarks, Llama decoder primitives, and a Cachemir
layer driver with encrypted KV reuse.

## Build

Use Linux x86-64, CMake 3.26+, a CUDA toolkit compatible with your GPU, GCC,
Git, Python 3, NTL, GMP, and C++11-ABI CPU LibTorch 2.6.0. The A100 profile
uses CUDA 12.4 and compute capability 80. Set `LIBTORCH_ROOT` to your
[LibTorch](https://pytorch.org/get-started/locally/) installation.

```sh
./scripts/build_phantom.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_PREFIX_PATH="$PWD/.deps/phantom;$LIBTORCH_ROOT"
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The dependency script pins the PhantomBootstrapping and PhantomFHE source
revisions and enables random sampling. `PHANTOM_PREFIX` overrides its output
directory. Set `CUDA_ARCHITECTURES` for the dependency build when changing GPU
architecture. CUDA benchmarks require an NVIDIA GPU and compatible driver.

## Run

```sh
build/bin/cipheropbench_bench --list
build/bin/cipheropbench_bench --method nexus --scope qk --warmup 1 --runs 3 --output bert.jsonl
build/bin/cipheropbench_bert_batch linear 2 1 3 batch.jsonl
CIPHEROPBENCH_DECODER_ONLY=1 build/bin/cipheropbench_decoder_bench rope_k 1 3 1 llama.jsonl
```

These commands use synthetic inputs. For BERT model traces, pass `--fixture DIR`
containing `hidden.bin`, `mask.bin`, and `layer0/` through the requested layer.
Files are row-major little-endian FP64 arrays. Tensor names and shapes are
defined in `src/fixtures.cpp`. Batch fixture lists contain one directory per
line. The operator batch executable prints its argument format when invoked
without arguments.

For the N=2^17 Cachemir profile, configure a separate build with
`-DCIPHEROPBENCH_UNIFIED_PHANTOM_GPU=ON` and the same dependency options. Run
`build-unified/bin/cipheropbench_cachemir_layer DIR` with a fresh output
directory on a GPU with at least 72 GiB free. For CPU checks alone, configure with
`-DCIPHEROPBENCH_CPU_ONLY=ON` and run CTest.

The package covers the shared operators and selected connected benchmarks.
Paper timing tables also use experiment-specific launchers, trace inputs,
and backend builds. See `licenses/` for upstream notices.
