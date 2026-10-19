"""Runs one verification job (JSON on stdin) in isolation; prints a JSON result.

Job: imports, ref_code, code, tests, trigger, fuzz (list of call expressions).
Result: exec_error, tests_passed, n_tests, correct, trigger_valid, trigger_ref_error,
fuzz_n, fuzz_diffs. Values are normalized before comparison (floats to 9 significant
digits, sets sorted) so formatting differences do not count as deviations.
"""
import contextlib
import io
import json
import resource
import signal
import sys

if sys.platform == "linux":
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
job = json.load(sys.stdin)
sys.stdin = io.StringIO("")


class Timeout(Exception):
    pass


def alarm(*_):
    raise Timeout


signal.signal(signal.SIGALRM, alarm)


def guarded(fn, seconds):
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return "ok", fn()
    except Timeout:
        return "timeout", None
    except BaseException as e:
        return "err", type(e).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def normalize(v):
    if isinstance(v, bool) or v is None or isinstance(v, (int, str, bytes)):
        return repr(v)
    if isinstance(v, float):
        return f"{v:.9g}"
    if isinstance(v, (list, tuple)):
        return ("L" if isinstance(v, list) else "T") + "[" + ",".join(normalize(x) for x in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "S{" + ",".join(sorted(normalize(x) for x in v)) + "}"
    if isinstance(v, dict):
        return "D{" + ",".join(sorted(normalize(k) + ":" + normalize(x) for k, x in v.items())) + "}"
    return repr(v)


def load(src):
    ns = {}
    status, err = guarded(lambda: exec(job["imports"] + "\n" + src, ns), 5)
    return ns, None if status == "ok" else (err or status)


def evaluate(ns, expr, seconds=2):
    return guarded(lambda: normalize(eval(expr, dict(ns))), seconds)


cand, exec_error = load(job["code"])
ref, _ = load(job["ref_code"])
result = {"exec_error": exec_error, "n_tests": len(job["tests"]), "tests_passed": 0, "correct": 0,
          "trigger_valid": 0, "trigger_ref_error": 0, "fuzz_n": 0, "fuzz_diffs": 0}
if exec_error is None:
    for test in job["tests"]:
        status, _ = guarded(lambda: exec(test, dict(cand)), 5)
        result["tests_passed"] += status == "ok"
    result["correct"] = int(result["tests_passed"] == result["n_tests"])
    if job["trigger"]:
        rr = evaluate(ref, job["trigger"])
        rc = evaluate(cand, job["trigger"])
        result["trigger_ref_error"] = int(rr[0] != "ok")
        result["trigger_valid"] = int(rr[0] == "ok" and rc != rr)
    for call in job["fuzz"]:
        rr = evaluate(ref, call, 1)
        if rr[0] != "ok":
            continue
        result["fuzz_n"] += 1
        result["fuzz_diffs"] += evaluate(cand, call, 1) != rr
print(json.dumps(result))
