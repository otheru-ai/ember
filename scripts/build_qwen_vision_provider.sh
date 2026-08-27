#!/usr/bin/env bash
# Build the lazy Qwen3.8 vision provider against the exact llama.cpp PR head
# used by Ember's converter/runtime audit. No model or mmproj weights are read.
set -euo pipefail

readonly llama_repo="https://github.com/ggml-org/llama.cpp.git"
readonly llama_ref="refs/pull/27774/head"
readonly llama_revision="abdc7a0bf815d3b83e26dd523c6960e4dd597e82"

usage() {
  echo "usage: $0 --build-dir DIR --install-dir DIR [--source-dir DIR] [--backend hip|cpu] [--jobs N] [--dry-run]" >&2
  exit 64
}

source_dir=""
build_dir=""
install_dir=""
backend="hip"
jobs="$(nproc)"
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir) [[ $# -ge 2 ]] || usage; source_dir="$2"; shift 2 ;;
    --build-dir) [[ $# -ge 2 ]] || usage; build_dir="$2"; shift 2 ;;
    --install-dir) [[ $# -ge 2 ]] || usage; install_dir="$2"; shift 2 ;;
    --backend) [[ $# -ge 2 ]] || usage; backend="$2"; shift 2 ;;
    --jobs) [[ $# -ge 2 ]] || usage; jobs="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) usage ;;
  esac
done
[[ -n "$build_dir" && -n "$install_dir" ]] || usage
[[ "$backend" == hip || "$backend" == cpu ]] || usage
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || usage

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ember_root="$(cd -- "$script_dir/.." && pwd -P)"
if [[ -z "$source_dir" ]]; then
  source_dir="$build_dir/llama.cpp"
fi

if [[ "$dry_run" == 1 ]]; then
  printf 'repository=%s\nref=%s\nrevision=%s\nbackend=%s\nsource=%s\nbuild=%s\ninstall=%s\n' \
    "$llama_repo" "$llama_ref" "$llama_revision" "$backend" \
    "$source_dir" "$build_dir/llama-build" "$install_dir"
  exit 0
fi

command -v git >/dev/null
command -v cmake >/dev/null
command -v "${CXX:-c++}" >/dev/null
command -v python3 >/dev/null
if [[ ! -e "$source_dir/.git" ]]; then
  [[ ! -e "$source_dir" ]] || {
    echo "source path exists but is not a git checkout: $source_dir" >&2
    exit 65
  }
  mkdir -p "$(dirname -- "$source_dir")"
  git clone --filter=blob:none --no-checkout "$llama_repo" "$source_dir"
fi
git -C "$source_dir" fetch --no-tags origin "$llama_ref"
[[ "$(git -C "$source_dir" rev-parse FETCH_HEAD)" == "$llama_revision" ]] || {
  echo "pinned llama.cpp ref moved away from $llama_revision" >&2
  exit 65
}
git -C "$source_dir" checkout --detach "$llama_revision"
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$llama_revision" ]] || exit 65

llama_build="$build_dir/llama-build"
cmake_args=(
  -S "$source_dir" -B "$llama_build"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$install_dir"
  -DBUILD_SHARED_LIBS=ON
  -DLLAMA_BUILD_COMMON=OFF
  -DLLAMA_BUILD_TOOLS=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_CURL=OFF
  -DLLAMA_TOOLS_INSTALL=OFF
  -DLLAMA_BUILD_MTMD=ON
  -DLLAMA_SUBPROCESS=OFF
  -DMTMD_VIDEO=OFF
)
if [[ "$backend" == hip ]]; then
  cmake_args+=(
    -DGGML_HIP=ON
    -DGGML_HIP_GRAPHS=OFF
    -DAMDGPU_TARGETS=gfx1151
  )
fi
cmake "${cmake_args[@]}"
cmake --build "$llama_build" --target mtmd -j"$jobs"
cmake --install "$llama_build"

adapter_build="$build_dir/adapter-build"
cmake -S "$ember_root/tools/qwen_vision_provider" -B "$adapter_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DEMBER_SOURCE_DIR="$ember_root" \
  -DLLAMA_CPP_SOURCE_DIR="$source_dir" \
  -DLLAMA_CPP_INSTALL_DIR="$install_dir"
cmake --build "$adapter_build" -j"$jobs"
cmake --install "$adapter_build"

lib_dir="$install_dir/lib"
[[ -d "$lib_dir" ]] || lib_dir="$install_dir/lib64"

python3 "$ember_root/scripts/check_qwen_vision_provider.py" \
  --provider "$lib_dir/libember_qwen4exp_vision_provider.so" \
  --require-llama-deps
printf '%s\n' "$llama_revision" > "$install_dir/LLAMA_CPP_REVISION"
echo "Qwen vision provider installed under $install_dir"
