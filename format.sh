#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -eo pipefail

# Prevent 'git rev-parse' from failing if we run this from inside the .git directory
# Reference: https://github.com/flashinfer-ai/flashinfer/blob/main/format.sh
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

yapf --in-place -r -p -e build -e third_party .
echo 'yapf done'

codespell eprof src include tests
echo 'codespell done'

ruff check eprof src include tests
echo 'ruff done'

find . -type f \( -name '*.c' -o -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.cuh' \) \
    -not -path "./third_party/*" \
    -not -path "./build/*" \
    -exec clang-format -i {} +
echo 'clang-format done'

if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    git --no-pager diff --name-only
    exit 1
fi

echo 'All checks passed'
