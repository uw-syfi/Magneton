#!/usr/bin/env bash
# Parse every C/C++ file the way clangd will, and fail on any diagnostic.
# Run from the repo root.
set -uo pipefail
fail=0
for f in lib/*/src/*.cpp lib/*/include/*.h \
         lib/eprof-kineto/libkineto/include/kineto_c_api.h \
         lib/eprof-kineto/libkineto/src/kineto_c_api.cpp; do
  out=$(timeout 240 clangd --check="$f" 2>&1)
  diag=$(grep -cE '^E\[[0-9:.]+\] \[[a-z_]+\] Line' <<<"$out")
  cfg=$(grep -c 'config error' <<<"$out")
  if [ "$diag" -ne 0 ] || [ "$cfg" -ne 0 ]; then
    printf 'FAIL %-44s diagnostics=%s config-errors=%s\n' "$f" "$diag" "$cfg"
    grep -E '^E\[[0-9:.]+\] \[[a-z_]+\] Line|config error' <<<"$out" | head -5
    fail=1
  fi
done
[ $fail -eq 0 ] && echo "clangd: all files clean"
exit $fail
