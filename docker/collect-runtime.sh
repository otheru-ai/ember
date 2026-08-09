#!/usr/bin/env bash
# Copy executable objects and their recursive dynamic-library closure while
# preserving absolute paths beneath a destination root. This keeps the release
# image independent of ROCm package-repository availability without copying the
# compiler, headers, profilers, or unrelated SDK libraries.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 DESTINATION ELF_OBJECT..." >&2
  exit 64
fi

destination="$1"
shift
mkdir -p "$destination"

declare -A seen=()
queue=("$@")

copy_object() {
  local object="$1"
  [[ -f "$object" ]] || {
    echo "runtime object not found: $object" >&2
    exit 66
  }
  local target="$destination$object"
  mkdir -p "$(dirname "$target")"
  cp -L --preserve=mode,timestamps "$object" "$target"
}

while [[ ${#queue[@]} -gt 0 ]]; do
  object="${queue[0]}"
  queue=("${queue[@]:1}")
  [[ -n "${seen[$object]:-}" ]] && continue
  seen["$object"]=1
  copy_object "$object"

  linkage="$(ldd "$object")"
  if grep -q 'not found' <<<"$linkage"; then
    echo "unresolved runtime dependency for $object:" >&2
    echo "$linkage" >&2
    exit 69
  fi

  while IFS= read -r dependency; do
    [[ -n "$dependency" && -z "${seen[$dependency]:-}" ]] &&
      queue+=("$dependency")
  done < <(
    awk '
      $2 == "=>" && $3 ~ /^\// { print $3 }
      $1 ~ /^\// { print $1 }
    ' <<<"$linkage"
  )
done

printf 'runtime closure: %d ELF object(s)\n' "${#seen[@]}"
