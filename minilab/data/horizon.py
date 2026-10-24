"""Long tasks: the toy code world's tasks, as long as we like (the horizon benchmark).

The coding eval's tasks take 1-6 tool calls. A long task grows with a parameter k:

    repair    one request, k bugs in a bigger project: run the tests, fix a function, run
              them again... until they all pass
    session   k requests in a row on the same project (explore, create, modify), each one
              made for the project as the previous ones left it

A long task's *length* is the number of tool calls its oracle makes, all requests together:
the reference solution's effort, like the human time of METR's tasks. Every long task is played
by its oracle when it's made (data/code.py play()), so it is solvable, and its length is known.

The same tasks are training data (SFT, the oracle's transcripts) and the benchmark, so the
benchmark keeps things training never shows: longer tasks (TRAIN_KS: up to 8 bugs and 12
requests; the benchmark goes to 16 and 32), its own ways of asking for a repair, and lead-ins
("Next, ...") on a session's later requests. On top of that, no training task starts like a
benchmark task (data/code.py eval_fingerprints).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

from minilab.data import code
from minilab.data.code import (MODULE_NAMES, Ctx, Module, Oracle, Project, Task, _asserts, _pick_funcs,
                               _python_ok, _run_output, _test_file, _traceback_line, body_bug, glob_paths, join,
                               tick)
from minilab.data.sandbox import Sandbox

FAMILIES = ["repair", "session"]
TRAIN_KS = {"repair": (2, 8), "session": (4, 12)}   # the k training draws from (both ends included)
# The kinds a session's requests are drawn from: the ones that take the project as it is
# (repairs need a bug planted first).
SESSION_KINDS = ["list_files", "find_def", "show_file", "explain", "run", "run_tests",
                 "create_func", "create_script", "rename", "change_const", "add_func"]


@dataclass
class Request:
    kind: str
    prompt: str
    check: Callable[[Sandbox, str], bool]
    calls: int   # the oracle's tool calls


@dataclass
class LongTask:
    family: str
    k: int
    files: dict[str, str]
    requests: list[Request]
    messages: list[dict]   # the oracle's transcript: the requests, tool calls, results and answers
    root: str              # the sandbox directory the transcript was played in

    @property
    def length(self) -> int:
        return sum(r.calls for r in self.requests)

    @property
    def fingerprint(self) -> str:
        return code.fingerprint(self.files, self.requests[0].prompt)


def _oracle_calls(messages: list[dict]) -> int:
    """Tool calls in an oracle transcript, without its shown mistakes (weight 0)."""
    return sum(1 for m in messages if m.get("tool_calls") and m.get("weight", 1) != 0)


# ---- repair: k bugs, one request ------------------------------------------------

REPAIR_PROMPTS = ["The tests fail. Fix all the bugs.", "Run the tests and fix every bug.",
                  "Several tests are failing. Fix them all.", "Make all the tests pass.",
                  "The tests catch some bugs in this project. Fix them all."]
EVAL_REPAIR_PROMPTS = ["Some tests don't pass. Can you fix the code?", "Get the test suite passing again.",
                       "Fix whatever is breaking the tests.", "The test files report failures: fix the code."]


def _big_project(rng: random.Random, n_modules: int) -> Project:
    """n modules of 2-3 functions, each with its test file, and maybe a README."""
    taken: set[str] = set()
    paths = [p for g in ["num", "text", "list", "mixed"] for p in MODULE_NAMES[g]]
    groups = {p: g for g in MODULE_NAMES for p in MODULE_NAMES[g]}
    modules = []
    for path in rng.sample(paths, n_modules):
        funcs = _pick_funcs(rng, groups[path], rng.choice([2, 3]), taken)
        if funcs:
            modules.append(Module(path, funcs))
    files = {m.path: m.text() for m in modules}
    tests = {}
    for m in modules:
        files[f"test_{m.path}"] = _test_file(rng, m)
        tests[f"test_{m.path}"] = m
    if rng.random() < 0.3:
        files["README.md"] = "# Tiny utils\n\nA small Python project.\n"
    order = list(files)
    rng.shuffle(order)
    return Project(modules, {p: files[p] for p in order}, tests=tests)


def _plant_bugs(rng: random.Random, p: Project, k: int) -> tuple[dict[str, str], dict[str, tuple[str, str]]] | None:
    """k bugs in k functions, each caught by its module's test with an AssertionError (not a crash
    in the module), each on a line of its own: (files, {function: (bad line, good line)})."""
    funcs = [f for f in p.funcs()]
    rng.shuffle(funcs)
    files, fixes = dict(p.files), {}
    for f in funcs:
        if len(fixes) == k:
            break
        bug = body_bug(rng, f)
        if bug is None:
            continue
        bad_body = bug[0]
        m = p.module_of(f.name)
        test = f"test_{m.path}"
        alone = dict(p.files)
        alone[m.path] = p.files[m.path].replace(f.code(), f.code(bad_body), 1)
        out = _run_output(alone, test)
        if "AssertionError" not in out or f'File "{m.path}"' in out:
            continue   # the test doesn't catch it, or it crashes inside the module
        bad_line, good_line = next((b, g) for b, g in zip(bad_body.split("\n"), f.kind.body.split("\n")) if b != g)
        bad_src, good_src = " " * 4 + bad_line, " " * 4 + good_line
        new_text = files[m.path].replace(f.code(), f.code(bad_body), 1)
        if new_text.count(bad_src) != 1:
            continue   # the edit needs a unique oldString
        files[m.path] = new_text
        fixes[f.name] = (bad_src, good_src)
    if len(fixes) < k or any(sum(t.count(bad) for t in files.values()) != 1 for bad, _ in fixes.values()):
        return None   # a later bug made an earlier one's line ambiguous
    return files, fixes


def repair_task(rng: random.Random, k: int, held_out: bool = False) -> Task | None:
    p = _big_project(rng, 1 + (k - 1) // 2)
    planted = _plant_bugs(rng, p, k)
    if planted is None:
        return None
    files, fixes = planted
    tests = sorted(p.tests)
    bugged = [(f, p.module_of(f.name)) for f in p.funcs() if f.name in fixes]
    prompt = rng.choice(EVAL_REPAIR_PROMPTS if held_out else REPAIR_PROMPTS)

    def solve(ctx: Ctx) -> Oracle:
        found = glob_paths(ctx, (yield "glob", {"pattern": "**/test_*.py"}))
        fixed = []
        for t in found:
            while True:
                out = ctx.rel((yield "shell", {"command": f"python3 {t}"}))
                if "Exited with code" not in out:
                    break
                _, _, src = _traceback_line(ctx, out)
                name = re.search(r"(\w+)\(", src.replace("assert ", "")).group(1)
                grep = ctx.rel((yield "grep", {"pattern": f"def {name}"}))
                path = grep.splitlines()[1].rstrip(":")
                yield "read", {"path": path}
                if name not in fixes or len(fixed) == len(fixes):
                    raise ValueError(f"the repair oracle is going around in circles on {name}")
                bad, good = fixes[name]
                yield "edit", {"path": path, "oldString": bad, "newString": good}
                fixed.append(f"{tick(name)} in {tick(path)}")
        return f"Fixed {len(fixed)} bug{'s' if len(fixed) > 1 else ''}: {join(fixed)}. All the tests pass now."

    def check(sb: Sandbox, answer: str) -> bool:
        now = sb.files()
        if any(now.get(t) != files[t] for t in tests):   # the tests themselves are untouched
            return False
        if any("Exited with code" in sb.shell(f"python3 {t}") for t in tests):
            return False
        # and the fixed functions are right on inputs the tests don't have
        return _python_ok(sb, "; ".join(f"from {m.import_name} import {f.name}; " + _asserts(f, random.Random(3))
                                        for f, m in bugged))

    return Task("repair", files, prompt, solve, check, "Fix failing tests")


def make_repair(rng: random.Random, k: int, held_out: bool = False, tries: int = 100) -> LongTask:
    for _ in range(tries):
        task = repair_task(rng, k, held_out)
        if task is None:
            continue
        try:
            with Sandbox(task.files) as sb:
                turns = code.play(task, sb, start=2)   # raises if the oracle fails its own check
                root = str(sb.root)
        except ValueError:
            continue
        messages = [{"role": "user", "content": task.prompt}, *turns]
        return LongTask("repair", k, task.files, [Request("repair", task.prompt, task.check, _oracle_calls(turns))],
                        messages, root)
    raise RuntimeError(f"could not make a repair task with {k} bugs")


# ---- session: k requests in a row -------------------------------------------------

def _next_request(rng: random.Random, sb: Sandbox, tries: int = 50) -> Task | None:
    """A request for the project as it is now (its metadata read back from its files)."""
    now = sb.files()
    p = code.project_from_files(now)
    for _ in range(tries):
        kind = rng.choice(SESSION_KINDS)
        try:
            task = code.TASKS[kind](rng, p)
        except (StopIteration, KeyError, ValueError, IndexError):
            continue
        if task is not None and task.files == now:   # not a project changed before the request (add_func's odd function)
            return task
    return None


EVAL_LEAD_INS = ["Next, ", "Now ", "OK, now ", "One more thing: "]   # the benchmark's only


def make_session(rng: random.Random, k: int, held_out: bool = False, tries: int = 100) -> LongTask:
    for _ in range(tries):
        start = code.make_project(rng).files
        requests, messages = [], []
        with Sandbox(start) as sb:
            for i in range(k):
                task = _next_request(rng, sb)
                if task is None:
                    break
                try:
                    turns = code.play(task, sb, start=len(messages) + 2)
                except ValueError:   # an oracle that fails its own check here: start over
                    break
                prompt = task.prompt
                if held_out and i > 0 and rng.random() < 0.5:
                    prompt = rng.choice(EVAL_LEAD_INS) + prompt[0].lower() + prompt[1:]
                requests.append(Request(task.kind, prompt, task.check, _oracle_calls(turns)))
                messages += [{"role": "user", "content": prompt}, *turns]
            root = str(sb.root)
        if len(requests) == k:
            return LongTask("session", k, start, requests, messages, root)
    raise RuntimeError(f"could not make a session of {k} requests")


MAKERS = {"repair": make_repair, "session": make_session}


def make_long_task(rng: random.Random, family: str, k: int, held_out: bool = False) -> LongTask:
    """A long task; held_out: the benchmark's version (its own phrasings)."""
    return MAKERS[family](rng, k, held_out)


def long_conversation(rng: random.Random, family: str) -> tuple[dict, str]:
    """A training conversation (data/code.py conversation()): a long task with k drawn from
    TRAIN_KS, its oracle's transcript typed like people do, at a plausible project path; and the
    task's fingerprint."""
    lo, hi = TRAIN_KS[family]
    task = make_long_task(rng, family, rng.randint(lo, hi))
    root = rng.choice(code.FAKE_ROOTS)
    messages = [code.system_message(root, rng)]
    for m in task.messages:
        messages.append({**m, "content": code.noisy(rng, m["content"])} if m["role"] == "user" else m)
    messages = code._relocate(messages, task.root, root)
    kind = "repair" if family == "repair" else "long_session"
    return {"kind": kind, "tools": code.tool_list(rng), "messages": messages}, task.fingerprint
