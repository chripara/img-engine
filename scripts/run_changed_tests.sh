#!/bin/bash
set -e

BASE="${1:-$(git merge-base main HEAD)}"
HEAD_REF="${2:-HEAD}"

CHANGED=$(git diff --name-only "$BASE" "$HEAD_REF" -- 'app/*.py')

TEST_FILES=""
for f in $CHANGED; do
  rel="${f#app/}"
  candidate="tests/$(dirname "$rel")/test_$(basename "$rel")"
  if [ -f "$candidate" ]; then
    TEST_FILES="$TEST_FILES $candidate"
  fi
done

if [ -n "$TEST_FILES" ]; then
  pytest -m "not gpu" $TEST_FILES
else
  echo "No matched test files for changed sources."
fi

pytest -m contract
