"""Check a spawned Modal call and print files from the penalty-shape volume.

usage: status.py CALL_ID            -> running / result
       status.py --file PATH        -> print a volume file (e.g. assay/x/summary.json)
       status.py --tail PATH [N]    -> last N lines of a volume file
"""
import sys

import modal

if sys.argv[1] in ("--file", "--tail"):
    vol = modal.Volume.from_name("penalty-shape")
    data = b"".join(vol.read_file(sys.argv[2])).decode(errors="replace")
    if sys.argv[1] == "--tail":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
        data = "\n".join(data.splitlines()[-n:])
    print(data)
else:
    fc = modal.FunctionCall.from_id(sys.argv[1])
    try:
        print("DONE", fc.get(timeout=0))
    except TimeoutError:
        print("RUNNING")
