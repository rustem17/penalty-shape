"""Block until a spawned Modal call finishes; print its result (or the exception)."""
import json
import sys

import modal

fc = modal.FunctionCall.from_id(sys.argv[1])
try:
    print(json.dumps(fc.get(timeout=float(sys.argv[2]) if len(sys.argv) > 2 else 7000), indent=1, default=str))
except Exception as e:
    print("FAILED", repr(e)[:3000])
