#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
prefix=${PHANTOM_PREFIX:-"$root/.deps/phantom"}
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
repo="$work/source"
git clone --quiet https://github.com/timzsu/Phantom_bootstrapping.git "$repo"
git -C "$repo" checkout --quiet ac9b82a5f667e91af62c8c82d8a3a7cd166cf5f4
git -C "$repo" submodule update --init --recursive
phantom="$repo/extern/phantom-fhe"
test "$(git -C "$phantom" rev-parse HEAD)" = f62cb99db97b5476adef09591596b1fb38c9eb99
git -C "$phantom" apply "$repo/cmake/phantom-fhe.patch"
git -C "$phantom" apply "$root/scripts/phantom-rns.patch"
python3 - "$repo" <<'PY'
from pathlib import Path
import sys

repo = Path(sys.argv[1])
prng = repo / "extern/phantom-fhe/src/prng.cu"
text = prng.read_text()
old = "out[start_pos + index] = barrett_reduce_uint64_uint64(1, mod.value(), mod.const_ratio()[1]);"
assert text.count(old) == 1
text = text.replace(old, old.replace("(1,", "(rnd[index],"))
old = "        auto start_pos = tid * 8;"
assert text.count(old) == 1
prng.write_text(text.replace(old, "        index = 0;\n        tries = 0;\n" + old))
header = repo / "include/ckks_evaluator.cuh"
text = header.read_text()
old = "    decryptor->decrypt(*context, ct, plain);\n  }\n"
assert text.count(old) == 1
header.write_text(text.replace(old, old + """
  inline void encrypt_symmetric(PhantomPlaintext &plain, PhantomCiphertext &ct) {
    decryptor->encrypt_symmetric(*context, plain, ct);
  }
"""))
PY
cmake -S "$repo" -B "$work/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-80}" \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DBUILD_EXAMPLES=OFF
cmake --build "$work/build" --parallel "${JOBS:-2}"
cmake --install "$work/build"
