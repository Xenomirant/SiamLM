#!/bin/bash
# Harbor verifier: runs in the task's container after the agent is done; writes /logs/verifier/reward.txt.
# Lines that contain "error" in any letter case: 3 (ERROR, Error, error).
expected=3
got=$(tr -d '[:space:]' < count.txt 2>/dev/null)
echo "expected: $expected, got: ${got:-<no count.txt>}"
if [ "$got" = "$expected" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
