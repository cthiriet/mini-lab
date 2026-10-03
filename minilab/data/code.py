"""The toy code world: tiny Python projects, coding tasks on them, and the agent transcripts
that solve them.

A *project* is a few small files: modules of little functions (`add`, `greet`, `reverse`...),
maybe a `main.py` that prints a few calls, a test file of asserts, a README. A *task* is a
request about a project ("Rename add to plus", "The tests fail, can you fix them?"), with:

- an oracle that solves it with opencode's tools, step by step (a generator that yields tool
  calls and receives their results), ending with the answer to the user;
- a check that tells whether any attempt solved it (the project's files after the attempt,
  and the final answer), used by the eval.

The four families of tasks:

    explore   list_files, find_def, show_file, explain, run, run_tests
    create    create_func, create_script
    modify    rename, change_const, add_func
    repair    fix_test, fix_crash (+ fix_distractor, the eval's fix_test with a lure in every module)
    + chat (small talk, identity, out of scope) and title (opencode's title requests)

Transcripts are played for real in a Sandbox (data/sandbox.py): every tool result is what
opencode would have sent, Python tracebacks included.
"""

from __future__ import annotations

import builtins
import dataclasses
import gzip
import hashlib
import json
import keyword
import multiprocessing
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Iterator

from minilab.data.conversations import NAME
from minilab.data.sandbox import OPENCODE_TOOLS, Sandbox

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Kind:
    """One kind of function: its names, parameters, body and what it does in English."""
    key: str
    names: tuple[str, ...]          # names it can get in a project
    params: tuple[str, ...]
    args: str                       # argument sampler, see sample_args()
    body: str                       # the body, indented relative to the def
    phrases: tuple[str, ...]        # "returns the sum of a and b"
    group: str                      # the module it lives in: "num", "text" or "list"
    bugs: tuple[tuple[str, str], ...] = ()   # (old, new): a substring of the body -> a bug


K = Kind
KINDS: list[Kind] = [
    # two numbers
    K("add", ("add", "plus", "sum_two"), ("a", "b"), "ii", "return a + b",
      ("returns the sum of a and b", "adds a and b"), "num", (("a + b", "a - b"), ("a + b", "a * b"), ("a + b", "a + a"))),
    K("subtract", ("subtract", "minus", "difference"), ("a", "b"), "ii", "return a - b",
      ("returns a minus b", "subtracts b from a"), "num", (("a - b", "a + b"), ("a - b", "b - a"))),
    K("multiply", ("multiply", "times", "product"), ("a", "b"), "ii", "return a * b",
      ("returns the product of a and b", "multiplies a by b"), "num", (("a * b", "a + b"), ("a * b", "a * a"))),
    K("larger", ("larger", "bigger", "max_of"), ("a", "b"), "ii", "if a > b:\n    return a\nreturn b",
      ("returns the larger of a and b", "returns the bigger of two numbers"), "num",
      (("a > b", "a < b"), ("return b", "return a"))),
    K("smaller", ("smaller", "min_of", "lower_of"), ("a", "b"), "ii", "if a < b:\n    return a\nreturn b",
      ("returns the smaller of a and b",), "num", (("a < b", "a > b"),)),
    K("average", ("average", "mean"), ("a", "b"), "ii", "return (a + b) / 2",
      ("returns the average of a and b",), "num", (("/ 2", "/ 3"), ("(a + b) / 2", "a + b / 2"))),
    K("power", ("power", "raise_to"), ("a", "b"), "pow", "return a ** b",
      ("returns a to the power of b",), "num", (("a ** b", "a * b"),)),
    K("divisible", ("is_divisible", "divides"), ("a", "b"), "div", "return a % b == 0",
      ("returns True if a is divisible by b",), "num", (("== 0", "== 1"),)),
    # one number
    K("square", ("square", "squared"), ("x",), "i", "return x * x",
      ("returns x squared", "returns the square of x"), "num", (("x * x", "x * 2"), ("x * x", "x + x"))),
    K("double", ("double", "twice"), ("x",), "i", "return x * 2",
      ("returns x doubled", "returns twice x"), "num", (("* 2", "* 3"), ("x * 2", "x + 2"))),
    K("half", ("half", "halve"), ("x",), "i", "return x // 2",
      ("returns half of x, rounded down",), "num", (("// 2", "// 3"), ("x // 2", "x * 2"))),
    K("increment", ("increment", "add_one", "next_number"), ("x",), "i", "return x + 1",
      ("returns x plus one", "adds one to x"), "num", (("+ 1", "+ 2"), ("x + 1", "x - 1"))),
    K("decrement", ("decrement", "sub_one", "previous"), ("x",), "i", "return x - 1",
      ("returns x minus one", "subtracts one from x"), "num", (("- 1", "- 2"), ("x - 1", "x + 1"))),
    K("negate", ("negate", "opposite"), ("x",), "i", "return -x",
      ("returns the opposite of x", "returns minus x"), "num", (("-x", "x"),)),
    K("cube", ("cube", "cubed"), ("x",), "i", "return x * x * x",
      ("returns x cubed", "returns the cube of x"), "num", (("x * x * x", "x * x"), ("x * x * x", "x * 3"))),
    K("is_even", ("is_even", "even"), ("x",), "i", "return x % 2 == 0",
      ("returns True if x is even", "checks whether x is even"), "num", (("== 0", "== 1"), ("% 2", "% 3"))),
    K("is_odd", ("is_odd", "odd"), ("x",), "i", "return x % 2 == 1",
      ("returns True if x is odd",), "num", (("== 1", "== 0"),)),
    K("is_positive", ("is_positive", "positive"), ("x",), "signed", "return x > 0",
      ("returns True if x is positive", "checks whether x is greater than zero"), "num",
      (("x > 0", "x < 0"), ("x > 0", "x > 5"))),
    K("absolute", ("absolute", "abs_value", "magnitude"), ("x",), "signed", "if x < 0:\n    return -x\nreturn x",
      ("returns the absolute value of x",), "num", (("x < 0", "x > 0"), ("return -x", "return x"))),
    K("factorial", ("factorial", "fact"), ("n",), "small",
      "result = 1\nfor i in range(1, n + 1):\n    result = result * i\nreturn result",
      ("returns the factorial of n", "multiplies all the numbers from 1 to n"), "num",
      (("n + 1", "n"), ("result = 1", "result = 0"), ("result * i", "result + i"))),
    K("sum_to", ("sum_to", "triangle", "sum_up_to"), ("n",), "i",
      "acc = 0\nfor i in range(n + 1):\n    acc = acc + i\nreturn acc",
      ("returns the sum of all the numbers from 0 to n", "adds up the numbers from 0 to n"), "num",
      (("n + 1", "n"), ("acc + i", "acc + 1"))),
    # text
    K("greet", ("greet", "hello", "welcome"), ("name",), "name", 'return "Hello, " + name + "!"',
      ("returns a greeting for name", "says hello to name"), "text",
      (('"Hello, "', '"Helo, "'), ('+ "!"', '+ "?"'), ('+ name +', '+ "name" +'))),
    K("shout", ("shout", "loud", "upper_case"), ("s",), "word", "return s.upper()",
      ("returns s in upper case", "makes s upper case"), "text", (("upper", "lower"), ("s.upper()", "s"))),
    K("whisper", ("whisper", "quiet", "lower_case"), ("s",), "loud", "return s.lower()",
      ("returns s in lower case",), "text", (("lower", "upper"),)),
    K("reverse", ("reverse", "backwards", "flip"), ("s",), "word", "return s[::-1]",
      ("returns s reversed", "reverses s"), "text", (("[::-1]", "[::1]"), ("[::-1]", "[1:]"))),
    K("length", ("length", "size", "num_chars"), ("s",), "word", "return len(s)",
      ("returns the number of characters in s", "returns the length of s"), "text",
      (("len(s)", "len(s) + 1"), ("len(s)", "len(s) - 1"))),
    K("first_letter", ("first_letter", "initial"), ("s",), "word", "return s[0]",
      ("returns the first letter of s",), "text", (("s[0]", "s[1]"), ("s[0]", "s[-1]"))),
    K("last_letter", ("last_letter", "final_letter"), ("s",), "word", "return s[-1]",
      ("returns the last letter of s",), "text", (("s[-1]", "s[0]"), ("s[-1]", "s[-2]"))),
    K("exclaim", ("exclaim", "excite"), ("s",), "word", 'return s + "!"',
      ("adds an exclamation mark to s", "returns s with an exclamation mark"), "text",
      (('"!"', '"?"'), ('s + "!"', '"!" + s'))),
    K("repeat", ("repeat", "echo"), ("s", "n"), "word_n", "return s * n",
      ("returns s repeated n times", "repeats s n times"), "text", (("s * n", "s * 2"), ("s * n", "s + s"))),
    K("count_vowels", ("count_vowels", "vowels"), ("s",), "word",
      'k = 0\nfor c in s:\n    if c in "aeiou":\n        k = k + 1\nreturn k',
      ("returns the number of vowels in s", "counts the vowels in s"), "text",
      (('"aeiou"', '"aeio"'), ("k + 1", "k + 2"), ("k = 0", "k = 1"))),
    K("join_words", ("join_words", "combine"), ("a", "b"), "two_words", 'return a + " " + b',
      ("joins a and b with a space", "returns a and b separated by a space"), "text",
      (('" "', '""'), ('a + " " + b', 'b + " " + a'))),
    # lists
    K("total", ("total", "sum_list", "add_all"), ("nums",), "list",
      "result = 0\nfor n in nums:\n    result = result + n\nreturn result",
      ("returns the sum of the numbers in nums", "adds up all the numbers in nums"), "list",
      (("result = 0", "result = 1"), ("result + n", "result + 1"))),
    K("largest", ("largest", "biggest", "max_item"), ("nums",), "list", "return max(nums)",
      ("returns the largest number in nums", "returns the biggest item of nums"), "list", (("max", "min"),)),
    K("smallest", ("smallest", "min_item"), ("nums",), "list", "return min(nums)",
      ("returns the smallest number in nums",), "list", (("min", "max"),)),
    K("count", ("count", "count_items", "how_many"), ("items",), "list", "return len(items)",
      ("returns the number of items", "counts the items"), "list", (("len(items)", "len(items) - 1"),)),
    K("first", ("first", "head"), ("items",), "list", "return items[0]",
      ("returns the first item", "returns the first element of items"), "list",
      (("[0]", "[1]"), ("[0]", "[-1]"))),
    K("last", ("last", "tail"), ("items",), "list", "return items[-1]",
      ("returns the last item", "returns the last element of items"), "list", (("[-1]", "[0]"),)),
    K("evens", ("evens", "only_even", "keep_even"), ("nums",), "list", "return [n for n in nums if n % 2 == 0]",
      ("returns the even numbers of nums", "keeps only the even numbers"), "list", (("== 0", "== 1"),)),
    K("doubled", ("doubled", "double_all"), ("nums",), "list", "return [n * 2 for n in nums]",
      ("returns every number of nums doubled", "doubles every number in nums"), "list",
      (("n * 2", "n + 2"), ("n * 2", "n * 3"))),
    K("contains", ("contains", "has_item", "includes"), ("items", "x"), "list_item", "return x in items",
      ("returns True if x is in items", "checks whether items contains x"), "list",
      (("x in items", "x not in items"),)),
    K("mean_list", ("mean_list", "average_list"), ("nums",), "list", "return sum(nums) / len(nums)",
      ("returns the average of nums",), "list", (("/ len(nums)", "/ 2"),)),
]
KIND = {k.key: k for k in KINDS}

NAMES = ["Ada", "Bob", "Sam", "Mia", "Leo", "Zoe", "Max", "Eva", "Tom", "Lia", "Ben", "Amy", "Kai", "Noa"]
WORDS = ["cat", "hello", "apple", "sun", "python", "robot", "tree", "moon", "code", "banana", "river", "star",
         "house", "music", "lemon", "tiger", "cloud", "pizza"]
MESSAGES = ["Hello, world!", "Hi there!", "Good morning", "I love Python", "Hello from prelude", "Welcome!",
            "Have a nice day", "Goodbye!", "Let's code", "It works!"]

# Identifiers used inside bodies: a function can't be named like one of them.
_BODY_WORDS = {"a", "b", "x", "n", "s", "c", "i", "k", "acc", "result", "items", "nums", "name", "upper", "lower",
               "len", "max", "min", "sum", "range", "print"}


def sample_args(rng: random.Random, kind: str) -> tuple:
    r = rng.randint
    return {
        "ii": lambda: (r(1, 12), r(1, 12)),
        "pow": lambda: (r(1, 5), r(0, 3)),
        "div": lambda: (r(1, 20), r(1, 5)),
        "i": lambda: (r(1, 12),),
        "signed": lambda: (r(-9, 12),),
        "small": lambda: (r(1, 6),),
        "name": lambda: (rng.choice(NAMES),),
        "word": lambda: (rng.choice(WORDS),),
        "loud": lambda: (rng.choice([w.upper() for w in WORDS] + [w.capitalize() for w in WORDS]),),
        "word_n": lambda: (rng.choice(WORDS), r(2, 3)),
        "two_words": lambda: (rng.choice(WORDS), rng.choice(WORDS)),
        "list": lambda: ([r(0, 9) for _ in range(r(3, 5))],),
        "list_item": lambda: ((lambda xs: (xs, rng.choice(xs) if rng.random() < 0.5 else r(0, 9)))([r(0, 9) for _ in range(r(3, 5))])),
    }[kind]()


def lit(v) -> str:
    """A Python literal, with double quotes for strings."""
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    return repr(v)


def indent(text: str, n: int = 4) -> str:
    return "\n".join((" " * n + line) if line else line for line in text.split("\n"))


@dataclass
class Func:
    kind: Kind
    name: str

    @property
    def sig(self) -> str:
        return f"{self.name}({', '.join(self.kind.params)})"

    def code(self, body: str | None = None) -> str:
        return f"def {self.sig}:\n{indent(body or self.kind.body)}\n"

    def call(self, args: tuple) -> str:
        return f"{self.name}({', '.join(lit(a) for a in args)})"

    def value(self, args: tuple, body: str | None = None):
        """What the function returns (the canonical body, or a buggy one)."""
        env: dict = {}
        exec(self.code(body), env)  # our own generated code
        return env[self.name](*args)


def body_bug(rng: random.Random, f: Func) -> tuple[str, str] | None:
    """(buggy body, the changed line of the good body) or None."""
    if not f.kind.bugs:
        return None
    old, new = rng.choice(f.kind.bugs)
    body = f.kind.body
    if old not in body:
        return None
    return body.replace(old, new, 1), old


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

MODULE_NAMES = {
    "num": ["calc.py", "math_utils.py", "number_utils.py", "ops.py", "arith.py", "mathlib.py"],
    "text": ["strings.py", "text.py", "words.py", "greetings.py", "textlib.py"],
    "list": ["lists.py", "stats.py", "items.py", "listlib.py"],
    "mixed": ["utils.py", "helpers.py", "tools.py", "lib.py", "core.py"],
}
SCRIPT_NAMES = ["main.py", "app.py", "hello.py", "script.py", "run.py", "demo.py"]
CONST_NAMES = {"name": ["NAME", "USER", "PLAYER"], "word": ["WORD", "TEXT"], "i": ["COUNT", "NUMBER", "SIZE", "LIMIT"],
               "list": ["NUMBERS", "SCORES", "VALUES"]}
PROJECT_TITLES = ["Calculator", "Word tools", "Tiny utils", "Playground", "Demo", "My project", "Toolbox", "Scratch"]
EXTRA_FILES = [
    ("README.md", 0.3, lambda rng: f"# {rng.choice(PROJECT_TITLES)}\n\nA small Python project.\n"),
    ("opencode.json", 0.2, lambda rng: '{\n  "$schema": "https://opencode.ai/config.json"\n}\n'),
    ("requirements.txt", 0.1, lambda rng: "pytest\n"),
    ("notes.txt", 0.08, lambda rng: rng.choice(["TODO: more tests\n", "Remember to run the tests.\n"])),
    ("config.json", 0.06, lambda rng: '{"debug": false}\n'),
    ("LICENSE", 0.06, lambda rng: "MIT License\n"),
]


@dataclass
class Module:
    path: str
    funcs: list[Func]

    def text(self, bodies: dict[str, str] | None = None) -> str:
        bodies = bodies or {}
        return "\n\n".join(f.code(bodies.get(f.name)) for f in self.funcs)

    @property
    def import_name(self) -> str:
        return self.path[:-3]


@dataclass
class Const:
    name: str      # NAME
    value: object  # "Ada"
    argtype: str   # which sample_args kind produced it


@dataclass
class Project:
    modules: list[Module]
    files: dict[str, str]                # path -> content, in creation order
    main: str | None = None              # the script that prints things, if any
    tests: dict[str, Module] = field(default_factory=dict)   # test file -> the module it tests
    const: Const | None = None           # a constant of `main`, if any
    calls: list[tuple[Func, tuple]] = field(default_factory=list)   # what `main` prints

    def funcs(self) -> list[Func]:
        return [f for m in self.modules for f in m.funcs]

    def module_of(self, name: str) -> Module:
        return next(m for m in self.modules if any(f.name == name for f in m.funcs))

    def names(self) -> set[str]:
        return {f.name for f in self.funcs()}


def _pick_funcs(rng: random.Random, group: str, n: int, taken: set[str]) -> list[Func]:
    kinds = [k for k in KINDS if group == "mixed" or k.group == group]
    out: list[Func] = []
    for kind in rng.sample(kinds, len(kinds)):
        if len(out) == n:
            break
        if any(f.kind.key == kind.key for f in out):
            continue
        name = rng.choice(kind.names)
        if _clashes(name, taken):
            continue
        taken.add(name)
        out.append(Func(kind, name))
    return out


def _clashes(name: str, taken: set[str]) -> bool:
    """Names stay greppable and renamable: none is a substring of another, or a body word."""
    return name in _BODY_WORDS or any(name in t or t in name for t in taken)


def _test_file(rng: random.Random, module: Module) -> str:
    lines = [f"from {module.import_name} import {', '.join(f.name for f in module.funcs)}", ""]
    per = 2 if len(module.funcs) <= 2 else 1
    for f in module.funcs:
        seen = set()
        for _ in range(per):
            args = sample_args(rng, f.kind.args)
            if lit(args) in seen:
                continue
            seen.add(lit(args))
            lines.append(f"assert {f.call(args)} == {lit(f.value(args))}")
    lines.append('print("all tests passed")')
    return "\n".join(lines) + "\n"


def make_project(rng: random.Random) -> Project:
    """A random little project: 1-2 modules, a script that uses them, maybe tests and a README."""
    taken: set[str] = set()
    groups = rng.sample(["num", "text", "list", "mixed"], rng.choice([1, 1, 2]))
    modules, used_paths = [], set()
    for g in groups:
        path = rng.choice([p for p in MODULE_NAMES[g] if p not in used_paths])
        used_paths.add(path)
        funcs = _pick_funcs(rng, g, rng.choice([1, 2, 2, 3]), taken)
        if funcs:
            modules.append(Module(path, funcs))
    proj = Project(modules, {})

    single = rng.random() < 0.25 and len(modules) == 1
    files: dict[str, str] = {}
    if not single:
        for m in modules:
            files[m.path] = m.text()
    # The script: a few prints of calls, one of them maybe through a constant.
    if single or rng.random() < 0.8:
        script = rng.choice(["main.py"] * 3 + SCRIPT_NAMES)
        chosen = rng.sample(proj.funcs(), min(len(proj.funcs()), rng.choice([1, 2, 2, 3])))
        imports, consts, prints = [], [], []
        if not single:
            for m in modules:
                names = [f.name for f in m.funcs if f in chosen]
                if names:
                    imports.append(f"from {m.import_name} import {', '.join(names)}")
        const_for = rng.choice(chosen) if rng.random() < 0.5 else None
        for f in chosen:
            args = sample_args(rng, f.kind.args)
            proj.calls.append((f, args))
            if f is const_for and f.kind.args in ("name", "word", "i", "list", "small", "signed"):
                ctype = {"small": "i", "signed": "i"}.get(f.kind.args, f.kind.args)
                proj.const = Const(rng.choice(CONST_NAMES[ctype]), args[0], f.kind.args)
                consts.append(f"{proj.const.name} = {lit(args[0])}")
                prints.append(f"print({f.name}({proj.const.name}))")
            else:
                prints.append(f"print({f.call(args)})")
        if single:
            text = modules[0].text() + "\n\n" + "\n".join(consts + prints) + "\n"
            modules[0].path = script  # the functions live in the script itself
        else:
            text = "\n".join(imports) + "\n\n" + "\n".join(consts + prints) + "\n"
        files[script] = text
        proj.main = script
    elif single:
        files[modules[0].path] = modules[0].text()
    # Tests of each module, most of the time.
    for m in modules:
        if not single and rng.random() < 0.65:
            path = f"test_{m.path}"
            files[path] = _test_file(rng, m)
            proj.tests[path] = m
    # Files that aren't Python, as in real projects (none has a word a function could be named).
    for path, prob, text in EXTRA_FILES:
        if rng.random() < prob:
            files[path] = text(rng)
    order = list(files)
    rng.shuffle(order)
    proj.files = {p: files[p] for p in order}
    return proj


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

Oracle = Generator[tuple, str, str]   # yields tool calls (name, args[, MISTAKE]), receives results, returns the answer
MISTAKE = "mistake"   # a call the transcript shows but SFT doesn't train on (weight 0): the model learns to recover


@dataclass
class Task:
    kind: str
    files: dict[str, str]
    prompt: str
    solve: Callable[["Ctx"], Oracle]
    check: Callable[[Sandbox, str], bool]
    title: str
    edits_project: bool = True


@dataclass
class Ctx:
    """What an oracle knows while solving: the sandbox root, to read tool results like the
    chat template shows them (paths relative to the project)."""
    root: str

    def rel(self, text: str) -> str:
        return text.replace(self.root + "/", "").replace(self.root, ".")


def join(items: list[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def tick(s: str) -> str:
    return f"`{s}`"


def glob_paths(ctx: Ctx, out: str) -> list[str]:
    return [] if out.startswith("No files") else ctx.rel(out).splitlines()


def _python_ok(sb: Sandbox, code: str) -> bool:
    out = sb.shell(f"python3 -c {json.dumps(code)}")
    return "Exited with code" not in out and "Error" not in out


def _asserts(f: Func, rng: random.Random, n: int = 3, name: str | None = None) -> str:
    lines = []
    for _ in range(n):
        args = sample_args(rng, f.kind.args)
        lines.append(f"assert {(name or f.name)}({', '.join(lit(a) for a in args)}) == {lit(f.value(args))}")
    return "; ".join(lines)


def _run_output(sb_files: dict[str, str], script: str) -> str:
    with Sandbox(sb_files) as sb:
        return sb.shell(f"python3 {script}")


def _traceback_line(ctx: Ctx, out: str) -> tuple[str, int, str]:
    """(file, line, source) of the innermost frame in a traceback."""
    frames = re.findall(r'File "([^"]+)", line (\d+), in \S+\n {4}(.*)\n', ctx.rel(out))
    path, line, src = frames[-1]
    return path, int(line), src.strip()


def _error_line(out: str) -> str:
    lines = [l for l in out.splitlines() if l.strip() and not l.startswith("Exited with code")]
    return lines[-1].strip()


# ---- explore ----------------------------------------------------------------

def task_list_files(rng: random.Random, p: Project) -> Task:
    py_only = rng.random() < 0.5 or all(f.endswith(".py") for f in p.files)
    prompt = rng.choice(["What files are in this project?", "List the files.", "Which files are here?",
                         "Show me the files in this folder.", "What's in this project?", "List all files."]
                        if not py_only else
                        ["List the Python files.", "Which Python files are in this project?",
                         "What Python files are here?", "Show me the Python files.", "Find all .py files."])
    pattern = "**/*.py" if py_only else "**/*"
    expected = [f for f in p.files if f.endswith(".py") or not py_only]

    def solve(ctx: Ctx) -> Oracle:
        paths = glob_paths(ctx, (yield "glob", {"pattern": pattern}))
        what = "Python file" if py_only else "file"
        n = len(paths)
        return f"This project has {n} {what}{'s' if n > 1 else ''}: {join([tick(x) for x in paths])}."

    def check(sb: Sandbox, answer: str) -> bool:
        return all(f in answer for f in expected)

    return Task("list_files", p.files, prompt, solve, check, "Project files overview" if not py_only
                else "Python files listing", edits_project=False)


def task_find_def(rng: random.Random, p: Project) -> Task:
    missing = rng.random() < 0.15
    if missing:
        name = rng.choice([n for k in KINDS for n in k.names if not _clashes(n, p.names())] or ["compute"])
    else:
        f = rng.choice(p.funcs())
        name = f.name
    prompt = rng.choice([f"Where is {name} defined?", f"Find the function {name}.", f"In which file is {name}?",
                         f"Where can I find {name}?", f"Find where {name} is defined.",
                         f"Which file defines {name}?"])
    where: dict = {}
    if not missing:
        path = next(path for path, text in p.files.items() if f"def {name}(" in text)
        line = p.files[path].splitlines().index(next(l for l in p.files[path].splitlines() if l.startswith(f"def {name}("))) + 1
        where = {"path": path, "line": line}

    def solve(ctx: Ctx) -> Oracle:
        out = ctx.rel((yield "grep", {"pattern": f"def {name}"}))
        if out.startswith("No matches"):
            return f"I couldn't find a function named {tick(name)} in this project."
        path = out.splitlines()[1].rstrip(":")
        line = int(re.search(r"Line (\d+):", out).group(1))
        return f"{tick(name)} is defined in {tick(path)}, line {line}."

    def check(sb: Sandbox, answer: str) -> bool:
        if missing:
            return "couldn't find" in answer or "no function" in answer.lower()
        return where["path"] in answer and re.search(rf"\b{where['line']}\b", answer) is not None

    return Task("find_def", p.files, prompt, solve, check, f"Locate {name} definition", edits_project=False)


def task_show_file(rng: random.Random, p: Project) -> Task:
    m = rng.choice(p.modules)
    path = m.path
    prompt = rng.choice([f"What's in {path}?", f"Show me {path}.", f"Read {path}.", f"What does {path} contain?",
                         f"Which functions are in {path}?", f"What functions does {path} define?",
                         f"Open {path}."])

    def solve(ctx: Ctx) -> Oracle:
        out = yield "read", {"path": path}
        sigs = re.findall(r"^\d+: def (\w+\([^)]*\)):", out, re.M)
        n = len(sigs)
        return f"{tick(path)} defines {n} function{'s' if n > 1 else ''}: {join([tick(s) for s in sigs])}."

    def check(sb: Sandbox, answer: str) -> bool:
        return all(f.name in answer for f in m.funcs)

    return Task("show_file", p.files, prompt, solve, check, f"Contents of {path}", edits_project=False)


def task_explain(rng: random.Random, p: Project) -> Task:
    f = rng.choice(p.funcs())
    m = p.module_of(f.name)
    phrase = rng.choice(f.kind.phrases)
    name_file = rng.random() < 0.4
    prompt = (rng.choice([f"What does {f.name} in {m.path} do?", f"Explain {f.name} from {m.path}.",
                          f"In {m.path}, what does {f.name} return?"]) if name_file else
              rng.choice([f"What does {f.name} do?", f"Explain the function {f.name}.", f"What does {f.name} return?",
                          f"How does {f.name} work?"]))

    def solve(ctx: Ctx) -> Oracle:
        path = m.path
        if not name_file:
            out = ctx.rel((yield "grep", {"pattern": f"def {f.name}"}))
            path = out.splitlines()[1].rstrip(":")
        yield "read", {"path": path}
        return f"{tick(f.sig)} {phrase}."

    def check(sb: Sandbox, answer: str) -> bool:
        return any(ph in answer for ph in f.kind.phrases)

    return Task("explain", p.files, prompt, solve, check, f"Explain {f.name} function", edits_project=False)


def task_run(rng: random.Random, p: Project) -> Task | None:
    if not p.main:
        return None
    script = p.main
    prompt = rng.choice([f"Run {script}.", f"What does {script} print?", f"Execute {script}.",
                         f"Can you run {script}?", f"What's the output of {script}?", f"Run {script} for me."]
                        + (["Run the project.", "Run the program."] if script == "main.py" else []))
    def solve(ctx: Ctx) -> Oracle:
        out = ctx.rel((yield "shell", {"command": f"python3 {script}"}))
        if "Exited with code" in out:
            return f"{tick(script)} crashed: {tick(_error_line(out))}"
        return f"{tick(script)} printed:\n```\n{out.rstrip()}\n```"

    def check(sb: Sandbox, answer: str) -> bool:
        expected = _run_output(p.files, script)
        if "Exited with code" in expected:
            return _error_line(expected).split(":")[0] in answer
        return all(line in answer for line in expected.splitlines())

    title = f"Run {script}" if script in prompt else "Run the project"
    return Task("run", p.files, prompt, solve, check, title, edits_project=False)


def task_run_tests(rng: random.Random, p: Project) -> Task | None:
    if not p.tests:
        return None
    named = len(p.tests) == 1 and rng.random() < 0.4
    tests = list(p.tests)
    prompt = (rng.choice([f"Run {tests[0]}.", f"Does {tests[0]} pass?", f"Run the tests in {tests[0]}."]) if named else
              rng.choice(["Run the tests.", "Do the tests pass?", "Are the tests passing?", "Check the tests.",
                          "Run the test suite.", "Can you run the tests?"]))
    def solve(ctx: Ctx) -> Oracle:
        found = tests if named else glob_paths(ctx, (yield "glob", {"pattern": "**/test_*.py"}))
        report = []
        for t in found:
            out = ctx.rel((yield "shell", {"command": f"python3 {t}"}))
            if "Exited with code" not in out:
                report.append(f"{tick(t)} passes.")
            elif "AssertionError" in out:
                _, line, src = _traceback_line(ctx, out)
                report.append(f"{tick(t)} fails on line {line}: {tick(src)}")
            else:
                report.append(f"{tick(t)} fails: {tick(_error_line(out))}")
        if all(r.endswith("passes.") for r in report):
            return "All tests pass." if len(report) > 1 else report[0].replace("passes.", "passes: all tests passed.")
        return " ".join(report)

    def check(sb: Sandbox, answer: str) -> bool:
        ok = all("Exited with code" not in _run_output(p.files, t) for t in tests)
        return ("pass" in answer and "fail" not in answer) if ok else "fail" in answer

    return Task("run_tests", p.files, prompt, solve, check, "Run the test suite", edits_project=False)


# ---- create -------------------------------------------------------------------

def _new_file_name(rng: random.Random, p: Project, group: str) -> str:
    options = [n for n in MODULE_NAMES[group] + MODULE_NAMES["mixed"] if n not in p.files]
    return rng.choice(options)


def task_create_func(rng: random.Random, p: Project) -> Task:
    group = rng.choice(["num", "text", "list"])
    path = _new_file_name(rng, p, group)
    funcs = _pick_funcs(rng, group, 2 if rng.random() < 0.2 else 1, set(p.names()))
    phrases = [rng.choice(f.kind.phrases) for f in funcs]
    if len(funcs) == 1:
        f, ph = funcs[0], phrases[0]
        prompt = rng.choice([f"Create {path} with a function {f.name} that {ph}.",
                             f"Write a function {f.sig} in {path} that {ph}.",
                             f"Make a new file {path} with {f.name}, which {ph}.",
                             f"Add a file {path} containing a function {f.name} that {ph}.",
                             f"Create a function {f.name} in {path}. It {ph}.",
                             f"In a new file {path}, write {f.name}: it {ph}."])
    else:
        (f1, f2), (p1, p2) = funcs, phrases
        prompt = rng.choice([f"Create {path} with two functions: {f1.name} that {p1}, and {f2.name} that {p2}.",
                             f"Write {path} with a function {f1.name} that {p1} and a function {f2.name} that {p2}."])
    module = Module(path, funcs)

    def solve(ctx: Ctx) -> Oracle:
        yield "write", {"path": path, "content": module.text()}
        return f"Created {tick(path)} with {join([tick(f.sig) for f in funcs])}."

    def check(sb: Sandbox, answer: str) -> bool:
        if sb.read_file(path) is None:
            return False
        code = f"from {module.import_name} import {', '.join(f.name for f in funcs)}; " + \
               "; ".join(_asserts(f, random.Random(1)) for f in funcs)
        return _python_ok(sb, code)

    title = f"Create {path} with {funcs[0].name}"
    return Task("create_func", p.files, prompt, solve, check, title)


def task_create_script(rng: random.Random, p: Project) -> Task:
    path = rng.choice([n for n in SCRIPT_NAMES + ["greeting.py", "hi.py", "say.py"] if n not in p.files])
    msg = rng.choice(MESSAGES + [f"Hello, {n}!" for n in NAMES])
    run = rng.random() < 0.4
    prompt = rng.choice([f"Create {path} that prints {msg}", f"Write a script {path} that prints \"{msg}\".",
                         f"Make {path} print \"{msg}\".", f"Create a file {path} which prints {msg}"])
    if run:
        prompt = prompt.rstrip(".") + rng.choice([", then run it.", " and run it.", ". Run it after.", f" and run {path}."])

    def solve(ctx: Ctx) -> Oracle:
        yield "write", {"path": path, "content": f"print({lit(msg)})\n"}
        if not run:
            return f"Created {tick(path)}."
        out = ctx.rel((yield "shell", {"command": f"python3 {path}"}))
        return f"Created {tick(path)}. It prints:\n```\n{out.rstrip()}\n```"

    def check(sb: Sandbox, answer: str) -> bool:
        return sb.read_file(path) is not None and sb.shell(f"python3 {path}").strip() == msg

    return Task("create_script", p.files, prompt, solve, check, f"Create {path} script")


# ---- modify -------------------------------------------------------------------

def _new_name(rng: random.Random, f: Func, taken: set[str]) -> str | None:
    """Another name of the same kind, or a variation. It may contain the old name (my_add):
    the rename replaces every whole-word occurrence anyway."""
    options = [n for n in f.kind.names if n != f.name] + [f"my_{f.name}", f"new_{f.name}", f"{f.name}_v2"]
    options = [n for n in options if not _clashes(n, taken - {f.name})]
    return rng.choice(options) if options else None


def _still_works(sb: Sandbox, p: Project, files: dict[str, str] | None = None) -> bool:
    """Scripts and tests that ran fine before (on `files`, by default the project's) still do."""
    for path in [p.main, *p.tests]:
        if path and "Exited with code" not in _run_output(files or p.files, path):
            if "Exited with code" in sb.shell(f"python3 {path}"):
                return False
    return True


def task_rename(rng: random.Random, p: Project) -> Task | None:
    f = rng.choice(p.funcs())
    g = _new_name(rng, f, p.names())
    if g is None:
        return None
    m = p.module_of(f.name)
    prompt = rng.choice([f"Rename {f.name} to {g}.", f"Rename the function {f.name} to {g}.",
                         f"Change the name of {f.name} to {g}.", f"Rename {f.name} to {g} everywhere.",
                         f"Call the function {g} instead of {f.name}.", f"Rename {f.name} in {m.path} to {g}."])
    users = [path for path, text in p.files.items() if re.search(rf"\b{f.name}\b", text)]
    # replaceAll changes every occurrence: skip names that a longer word holds (`from greetings import
    # greet`) or that a string uses (`lower_case("HELLO") == "hello"` for hello)
    if any(re.search(rf"\w{f.name}|{f.name}\w", text) for text in p.files.values()) or \
            any(re.search(rf"[\"'][^\"'\n]*\b{f.name}\b", text) for path, text in p.files.items() if path.endswith(".py")):
        return None

    def solve(ctx: Ctx) -> Oracle:
        out = ctx.rel((yield "grep", {"pattern": f.name}))
        paths = [l.rstrip(":") for l in out.splitlines() if l and not l.startswith(("Found", " "))]
        for path in paths:
            yield "edit", {"path": path, "oldString": f.name, "newString": g, "replaceAll": True}
        return f"Renamed {tick(f.name)} to {tick(g)} in {join([tick(x) for x in paths])}."

    def check(sb: Sandbox, answer: str) -> bool:
        files = sb.files()
        if any(re.search(rf"\b{f.name}\b", files.get(path, "")) for path in users):
            return False
        mod = files.get(m.path, "")
        return f"def {g}(" in mod and _still_works(sb, p)

    return Task("rename", p.files, prompt, solve, check, f"Rename {f.name} to {g}")


def task_change_const(rng: random.Random, p: Project) -> Task | None:
    c = p.const
    if c is None or not p.main:
        return None
    new = sample_args(rng, c.argtype)[0]
    for _ in range(5):
        if new != c.value:
            break
        new = sample_args(rng, c.argtype)[0]
    if new == c.value:
        return None
    script = p.main
    name_file = rng.random() < 0.6
    shown = new if isinstance(new, str) else lit(new)
    prompt = rng.choice([f"Change {c.name} to {shown}", f"Set {c.name} to {lit(new)}", f"Make {c.name} equal to {lit(new)}",
                         f"Use {shown} for {c.name}"])
    prompt += f" in {script}." if name_file else "."
    run = rng.random() < 0.35
    if run:
        prompt = prompt[:-1] + rng.choice([", then run it.", " and run the script.", f" and run {script}.",
                                           f", then run {script}."])
    old_line, new_line = f"{c.name} = {lit(c.value)}", f"{c.name} = {lit(new)}"

    def solve(ctx: Ctx) -> Oracle:
        path = script
        if not name_file:
            out = ctx.rel((yield "grep", {"pattern": f"{c.name} ="}))
            path = out.splitlines()[1].rstrip(":")
        yield "read", {"path": path}
        yield from _recovering_edit(rng, path, old_line, new_line)
        answer = f"Set {tick(c.name)} to {tick(lit(new))} in {tick(path)}."
        if run:
            out = ctx.rel((yield "shell", {"command": f"python3 {path}"}))
            answer += f" It prints:\n```\n{out.rstrip()}\n```"
        return answer

    def check(sb: Sandbox, answer: str) -> bool:
        lines = (sb.read_file(script) or "").splitlines()   # whole lines: "N = 1" is a piece of "N = 12"
        return new_line in lines and old_line not in lines

    return Task("change_const", p.files, prompt, solve, check, f"Set {c.name} to {shown}")


def _unique_tail(text: str) -> str:
    """The fewest last lines of `text` found only once in it: an oldString to append after."""
    lines = text.rstrip("\n").split("\n")
    for k in range(1, len(lines) + 1):
        tail = "\n".join(lines[-k:])
        if text.count(tail) == 1:
            return tail
    return text.rstrip("\n")


def task_add_func(rng: random.Random, p: Project) -> Task | None:
    m = rng.choice([m for m in p.modules if m.path in p.files] or [None])
    if m is None:
        return None
    group = m.funcs[0].kind.group
    new = _pick_funcs(rng, group if group != "mixed" else rng.choice(["num", "text", "list"]), 1, set(p.names()))
    if not new:
        return None
    g = new[0]
    ph = rng.choice(g.kind.phrases)
    # A third of the time, a function of the file isn't written the usual way (here, it has a bug):
    # the file must keep it as it is.
    files = dict(p.files)
    odd = rng.choice(m.funcs) if rng.random() < 0.35 else None
    bug = body_bug(rng, odd) if odd else None
    if bug and m.path in files:
        files[m.path] = files[m.path].replace(odd.code(), odd.code(bug[0]), 1)
    old_text = files[m.path]
    prompt = rng.choice([f"Add a function {g.name} to {m.path} that {ph}.", f"In {m.path}, add {g.sig}, which {ph}.",
                         f"Add {g.name} to {m.path}: it {ph}.", f"Add a function {g.sig} to {m.path}. It {ph}.",
                         f"Write a new function {g.name} in {m.path} that {ph}."])

    def solve(ctx: Ctx) -> Oracle:
        # The new function goes after the file's last lines with an edit: rewriting the whole file
        # with `write` made the model copy it back, and a tiny model garbles long copies.
        out = yield "read", {"path": m.path}
        tail = _unique_tail("\n".join(re.sub(r"^\d+: ", "", l) for l in out.splitlines()[1:]))
        yield "edit", {"path": m.path, "oldString": tail, "newString": tail + "\n\n\n" + g.code().rstrip("\n")}
        return f"Added {tick(g.sig)} to {tick(m.path)}."

    def check(sb: Sandbox, answer: str) -> bool:
        code = f"from {m.import_name} import {g.name}; " + _asserts(g, random.Random(2))
        kept = old_text.rstrip() in (sb.read_file(m.path) or "")   # everything else, exactly as it was
        return kept and _python_ok(sb, code) and _still_works(sb, p, files)

    return Task("add_func", files, prompt, solve, check, f"Add {g.name} to {m.path}")


# ---- repair -------------------------------------------------------------------

def task_fix_test(rng: random.Random, p: Project, distractor_frac: float = 0.8) -> Task | None:
    """A function has a bug that its test catches: run the test, find the function, fix it.
    Most broken modules also have a distractor, a function whose correct code looks like one of
    the bug's (see _distractor)."""
    if not p.tests:
        return None
    test = rng.choice(list(p.tests))
    m = p.tests[test]
    f = rng.choice(m.funcs)
    bug = body_bug(rng, f)
    if bug is None:
        return None
    bad_body, good_part = bug
    module = Module(m.path, list(m.funcs))
    distractor = _distractor(rng, f, bad_body, p) if rng.random() < distractor_frac else None
    if distractor_frac == 1 and distractor is None:
        return None
    if distractor:
        module.funcs.insert(rng.randint(0, len(module.funcs)), distractor)
    files = dict(p.files)
    files[m.path] = module.text({f.name: bad_body})
    bad_line, good_line = next((b, g) for b, g in zip(bad_body.split("\n"), f.kind.body.split("\n")) if b != g)
    bad_src, good_src = " " * 4 + bad_line, " " * 4 + good_line
    if files[m.path].count(bad_src) != 1:
        return None  # the edit needs a unique oldString: keep to bugs on a line of their own
    out = _run_output(files, test)
    if "AssertionError" not in out or f'File "{m.path}"' in out:
        return None  # the test doesn't catch this bug, or crashes inside the module
    who = rng.random()
    if who < 0.35:
        prompt = rng.choice(["The tests fail. Can you fix them?", "Fix the failing test.", "The tests are failing, fix the bug.",
                             "Run the tests and fix any bug.", "Something is broken, the tests fail. Fix it."])
        known_test = False
    elif who < 0.7:
        prompt = rng.choice([f"{test} fails, please fix it.", f"Fix the bug caught by {test}.",
                             f"{test} is failing. Can you fix the code?", f"Make {test} pass."])
        known_test = True
    else:
        prompt = rng.choice([f"There is a bug in {f.name}. Fix it and run {test}.",
                             f"Fix the bug in {f.name}, then check with {test}.",
                             f"{f.name} is wrong, fix it. The test is {test}."])
        known_test = True

    def solve(ctx: Ctx) -> Oracle:
        tests = [test] if known_test else glob_paths(ctx, (yield "glob", {"pattern": "**/test_*.py"}))
        failing, out = None, ""
        for t in tests:
            out = ctx.rel((yield "shell", {"command": f"python3 {t}"}))
            if "Exited with code" in out:
                failing = t
                break
        _, _, src = _traceback_line(ctx, out)
        name = re.search(r"(\w+)\(", src.replace("assert ", "")).group(1) if src.startswith("assert") else f.name
        grep = ctx.rel((yield "grep", {"pattern": f"def {name}"}))
        path = grep.splitlines()[1].rstrip(":")
        yield "read", {"path": path}
        yield from _recovering_edit(rng, path, bad_src, good_src)
        after = ctx.rel((yield "shell", {"command": f"python3 {failing}"}))
        done = "Exited with code" not in after
        return (f"Fixed {tick(name)} in {tick(path)}: it used {tick(bad_line.strip())} instead of "
                f"{tick(good_line.strip())}." + (" The tests pass now." if done else ""))

    def check(sb: Sandbox, answer: str) -> bool:
        return all("Exited with code" not in sb.shell(f"python3 {t}") for t in p.tests) and _still_works(sb, p)

    title = (f"Fix bug in {f.name}" if f.name in prompt else f"Fix failing {test}" if test in prompt
             else "Fix failing tests")   # what the request says: a title can't know more
    return Task("fix_test", files, prompt, solve, check, title)


def bad_lines(f: Func, bad_body: str) -> set[str]:
    """The lines of f's *other* bugs: what a bug in f could look like, besides the one it has."""
    lines = set()
    for old, new in f.kind.bugs:
        other = f.kind.body.replace(old, new, 1)
        if other != bad_body:
            lines |= set(other.split("\n")) - set(f.kind.body.split("\n"))
    return lines


def _distractor(rng: random.Random, f: Func, bad_body: str, p: Project) -> Func | None:
    """A function whose correct code is one of f's *other* bugs (multiply's `return a * b` next to
    a broken add): the fix must go to the line of the failing function, not to what looks like a bug."""
    lines = bad_lines(f, bad_body)
    kinds = [k for k in KINDS if k.key != f.kind.key and lines & set(k.body.split("\n"))]
    for kind in rng.sample(kinds, len(kinds)):
        name = rng.choice(kind.names)
        if not _clashes(name, p.names()):
            return Func(kind, name)
    return None


def task_fix_crash(rng: random.Random, p: Project) -> Task | None:
    """The script calls a function by a misspelled name: run it, read it, fix the typo."""
    if not p.main or not p.calls:
        return None
    script = p.main
    f, _ = rng.choice(p.calls)
    typo = _typo(rng, f.name)
    if typo is None or re.search(rf"\b{typo}\b", "".join(p.files.values())):
        return None
    text = p.files[script]
    call_line = next((l for l in text.splitlines() if l.startswith("print(") and f"{f.name}(" in l), None)
    if call_line is None:
        return None
    bad = call_line.replace(f"{f.name}(", f"{typo}(", 1)
    files = dict(p.files)
    files[script] = text.replace(call_line, bad, 1)
    good_out = _run_output(p.files, script)
    if "Exited with code" in good_out or "NameError" not in _run_output(files, script):
        return None
    prompt = rng.choice([f"{script} crashes, can you fix it?", f"Fix {script}.", f"{script} doesn't work. Fix it.",
                         f"Run {script} and fix the error.", f"There's an error in {script}, please fix it.",
                         f"Why does {script} fail? Fix it."])

    def solve(ctx: Ctx) -> Oracle:
        out = ctx.rel((yield "shell", {"command": f"python3 {script}"}))
        _, _, src = _traceback_line(ctx, out)
        yield "read", {"path": script}
        fixed = src.replace(f"{typo}(", f"{f.name}(", 1)
        yield from _recovering_edit(rng, script, src, fixed)
        after = ctx.rel((yield "shell", {"command": f"python3 {script}"}))
        return (f"Fixed the typo in {tick(script)}: {tick(typo)} should be {tick(f.name)}. "
                f"It now prints:\n```\n{after.rstrip()}\n```")

    def check(sb: Sandbox, answer: str) -> bool:
        return sb.shell(f"python3 {script}") == good_out

    return Task("fix_crash", files, prompt, solve, check, f"Fix {script} crash")


def _slip(rng: random.Random, old: str) -> str:
    """oldString copied slightly wrong: an extra space, or a dropped character. Never a piece of
    the line (a missing indent space, a dropped last character): that edit would go through, into
    a broken file."""
    for _ in range(10):
        if rng.random() < 0.4:
            slip = " " + old if not old.startswith(" ") else old.replace(" ", "  ", 1)
        else:
            i = rng.randrange(len(old.strip()) or 1) + len(old) - len(old.lstrip())
            slip = old[:i] + old[i + 1:]
        if slip not in old:
            return slip
    return old + "x"


def _recovering_edit(rng: random.Random, path: str, old: str, new: str) -> Oracle:
    """Now and then a first edit whose oldString doesn't match (not trained on), then the file is
    read again and the edit made right: the model learns what to do after "Could not find oldString"."""
    if rng.random() < 0.15:
        yield "edit", {"path": path, "oldString": _slip(rng, old), "newString": new}, MISTAKE
        yield "read", {"path": path}
    yield "edit", {"path": path, "oldString": old, "newString": new}
    return ""


def _typo(rng: random.Random, name: str) -> str | None:
    if len(name) < 4:
        return None
    i = rng.randrange(1, len(name) - 1)
    ops = [name[:i] + name[i + 1:],                      # drop a letter
           name[:i] + name[i] + name[i:],                # double a letter
           name[:i - 1] + name[i] + name[i - 1] + name[i + 1:]]   # swap two letters
    typo = rng.choice(ops)
    ok = typo != name and typo.isidentifier() and not keyword.iskeyword(typo) and not hasattr(builtins, typo)
    return typo if ok else None


# ---- chat ---------------------------------------------------------------------

CAPABILITIES = "I can list, read, run, create, edit and fix the Python files in this project."
GREETINGS = ["hi", "hello", "hey", "Hi!", "Hello!", "Hey there", "good morning", "yo", "Hi, how are you?"]
IDENTITY = ["Who are you?", "What are you?", "What's your name?", "Are you a human?", "Which model are you?",
            "Tell me about yourself."]
ABILITIES = ["What can you do?", "How can you help me?", "What are you able to do?", "help"]
THANKS = ["thanks", "Thank you!", "thx", "Great, thanks!", "Perfect, thank you.", "nice"]
OUT_OF_SCOPE = ["Write me a poem.", "What's the weather today?", "Tell me a joke.", "Who won the World Cup?",
                "Build me a website.", "Translate hello to French.", "What is the capital of France?",
                "Explain quantum physics.", "Book a flight to Paris.", "Write a React app.", "Deploy this to AWS.",
                "What time is it?", "Write a Rust program."]


def chat_turn(rng: random.Random) -> tuple[str, str, str]:
    """(kind, prompt, answer) of a message that needs no tool."""
    r = rng.random()
    if r < 0.25:
        return "greeting", rng.choice(GREETINGS), rng.choice(["Hi!", "Hello!", "Hey!"]) + " " + \
            rng.choice(["What should we do in this project?", CAPABILITIES, "How can I help with your code?"])
    if r < 0.45:
        return "identity", rng.choice(IDENTITY), (f"I'm {NAME}, a tiny language model trained from scratch on a "
                                                  "laptop. " + CAPABILITIES)
    if r < 0.6:
        return "abilities", rng.choice(ABILITIES), CAPABILITIES + " Try \"Run the tests\" or \"Rename add to plus\"."
    if r < 0.75:
        return "thanks", rng.choice(THANKS), rng.choice(["You're welcome!", "Happy to help!", "Anytime!"])
    return "out_of_scope", rng.choice(OUT_OF_SCOPE), ("Sorry, I can't do that: I'm a tiny model that only knows small "
                                                      "Python projects. " + CAPABILITIES)


# ---------------------------------------------------------------------------
# Sampling tasks and playing them
# ---------------------------------------------------------------------------

def _renamed(task: Task | None, kind: str) -> Task | None:
    return dataclasses.replace(task, kind=kind) if task else None


TASKS: dict[str, Callable[[random.Random, Project], Task | None]] = {
    "list_files": task_list_files, "find_def": task_find_def, "show_file": task_show_file, "explain": task_explain,
    "run": task_run, "run_tests": task_run_tests,
    "create_func": task_create_func, "create_script": task_create_script,
    "rename": task_rename, "change_const": task_change_const, "add_func": task_add_func,
    "fix_test": task_fix_test, "fix_crash": task_fix_crash,
}
# The eval's kinds: also fix_test with a distractor in every broken module, like opencode's demo project.
EVAL_TASKS = {**TASKS, "fix_distractor": lambda rng, p: _renamed(task_fix_test(rng, p, distractor_frac=1),
                                                                "fix_distractor")}
FAMILIES = {
    "explore": ["list_files", "find_def", "show_file", "explain", "run", "run_tests"],
    "create": ["create_func", "create_script"],
    "modify": ["rename", "change_const", "add_func"],
    "repair": ["fix_test", "fix_crash", "fix_distractor"],
}
READ_ONLY = {"list_files", "find_def", "show_file", "explain", "run", "run_tests"}


def sample_task(rng: random.Random, kind: str, tries: int = 300) -> tuple[Task, Project]:
    """A task of this kind on a fresh random project (resampled until the kind applies)."""
    for _ in range(tries):
        project = make_project(rng)
        task = EVAL_TASKS[kind](rng, project)
        if task is not None:
            return task, project
    raise RuntimeError(f"could not make a {kind} task")


def make_task(rng: random.Random, kind: str) -> Task:
    return sample_task(rng, kind)[0]


def noisy(rng: random.Random, prompt: str, chat: bool = False) -> str:
    """How people type: lower case, no final period, "please", and `opencode run`'s quotes."""
    if rng.random() < 0.15:
        prompt = prompt[0].lower() + prompt[1:]
    if rng.random() < 0.2 and prompt.endswith("."):
        prompt = prompt[:-1]
    r = rng.random()
    if not chat and r < 0.06:
        prompt = "Please " + prompt[0].lower() + prompt[1:]
    elif not chat and r < 0.1 and not prompt.endswith("?"):
        prompt = prompt.rstrip(".") + " please"
    if rng.random() < 0.2:
        prompt = f'"{prompt}"'   # `opencode run "..."` sends the message in quotes
    return prompt


FAKE_ROOTS = ["/home/user/project", "/Users/alex/code/demo", "/Users/sam/projects/app", "/home/dev/src/tool",
              "/Users/me/work/scratch"]
SYSTEM = ("You are an AI agent running in OpenCode, a coding agent harness. Help the user accomplish their goals "
          "using the tools you have available.\n\n<env>\n  Working directory: {root}\n  Workspace root folder: {root}\n"
          "  Is directory a git repo: yes\n  Platform: {platform}\n</env>")
TITLE_SYSTEM = "You are a title generator. You output ONLY a thread title. Nothing else."


def system_message(root: str, rng: random.Random) -> dict:
    return {"role": "system", "content": SYSTEM.format(root=root, platform=rng.choice(["darwin", "linux"]))}


def tool_list(rng: random.Random) -> list[str]:
    """What opencode offers: always the same list, with now and then an extra (MCP) tool or two."""
    tools = list(OPENCODE_TOOLS)
    if rng.random() < 0.1:
        tools += rng.sample(["todowrite", "lsp", "context7_query", "github_search", "browser", "memory"], rng.randint(1, 2))
    return tools


def play(task: Task, sb: Sandbox, start: int = 0) -> list[dict]:
    """Run the task's oracle in the sandbox: the assistant and tool messages after the prompt."""
    ctx = Ctx(str(sb.root))
    oracle = task.solve(ctx)
    messages: list[dict] = []
    i = start
    try:
        call = next(oracle)
        while True:
            name, args = call[:2]
            call_id = f"call_{i}"
            i += 1
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"id": call_id, "type": "function",
                                             "function": {"name": name, "arguments": json.dumps(args)}}]})
            if call[2:] == (MISTAKE,):
                messages[-1]["weight"] = 0   # OpenAI's fine-tuning format: shown, not trained on
            messages.append({"role": "tool", "tool_call_id": call_id, "content": sb.call(name, args)})
            call = oracle.send(messages[-1]["content"])
    except StopIteration as stop:
        messages.append({"role": "assistant", "content": stop.value})
    # A transcript whose oracle didn't solve its task would teach the model to stop at a broken
    # project, or to say it fixed what it didn't.
    if not task.check(sb, messages[-1]["content"]):
        raise ValueError(f"the {task.kind} oracle failed its own check: {messages[-1]['content']!r}")
    return messages


def _relocate(messages: list[dict], src: str, dst: str) -> list[dict]:
    """Move a transcript from the sandbox's temporary directory to a plausible project path."""
    return json.loads(json.dumps(messages).replace(json.dumps(src)[1:-1], json.dumps(dst)[1:-1]))


CHAT_TITLES = {"greeting": "Greeting", "identity": "Assistant identity question", "abilities": "Assistant capabilities",
               "thanks": "Thanks", "out_of_scope": "Off-topic request"}


def conversation(rng: random.Random, kind: str) -> dict:
    """One training conversation of the given kind, in OpenAI's message format.

    kind: a task kind, "chat", "title" (opencode's request for a session title), or "session":
    several requests in a row on the same project. Either a first look at it, then a task (maybe
    fixing what the look found), or a task, then one or two more on the project as it now is."""
    root = rng.choice(FAKE_ROOTS)
    if kind == "title":
        if rng.random() < 0.15:
            ck, prompt, _ = chat_turn(rng)
            title = CHAT_TITLES[ck]
        else:
            task = make_task(rng, rng.choice(list(TASKS)))
            prompt, title = noisy(rng, task.prompt), task.title
        return {"kind": "title", "tools": None,
                "messages": [{"role": "system", "content": TITLE_SYSTEM}, {"role": "user", "content": prompt},
                             {"role": "assistant", "content": title}]}
    if kind == "chat":
        _, prompt, answer = chat_turn(rng)
        return {"kind": "chat", "tools": tool_list(rng),
                "messages": [system_message(root, rng), {"role": "user", "content": noisy(rng, prompt, chat=True)},
                             {"role": "assistant", "content": answer}]}

    messages = [system_message(root, rng)]
    if kind != "session":
        task = make_task(rng, kind)
        with Sandbox(task.files) as sb:
            messages.append({"role": "user", "content": noisy(rng, task.prompt)})
            messages += play(task, sb, start=len(messages))
            _maybe_small_talk(rng, messages)
            messages = _relocate(messages, str(sb.root), root)
        return {"kind": kind, "tools": tool_list(rng), "messages": messages}

    last, project = sample_task(rng, rng.choice(list(TASKS)))
    if rng.random() < 0.5:
        # A first look at the project as the second request finds it (e.g. the failing test, then "fix it").
        seen = dataclasses.replace(project, files=last.files)
        first = next((t for k in rng.sample(sorted(READ_ONLY - {"explain"}), len(READ_ONLY) - 1)
                      if (t := TASKS[k](rng, seen)) is not None), None)
        plan = [first, last] if first else [last]
    else:
        # Any request, then one or two more on the project as it now is (after a rename, a fix...).
        plan = [last, None] + ([None] if rng.random() < 0.4 else [])
    with Sandbox(plan[0].files) as sb:
        for task in plan:
            follow_up = task is None
            if follow_up:
                task = _follow_up(rng, dataclasses.replace(project, files=sb.files()))
                if task is None:
                    break
            prompt = {"role": "user", "content": noisy(rng, task.prompt)}
            try:
                turns = play(task, sb, start=len(messages) + 1)
            except ValueError:
                if not follow_up:
                    raise
                break  # its check reads the stale metadata (a function renamed earlier): end the session here
            messages += [prompt, *turns]
        _maybe_small_talk(rng, messages)
        messages = _relocate(messages, str(sb.root), root)
    return {"kind": kind, "tools": tool_list(rng), "messages": messages}


FOLLOW_UPS = ["run", "run_tests", "list_files", "show_file", "find_def", "create_func", "create_script"]


def _follow_up(rng: random.Random, p: Project) -> Task | None:
    """A request whose oracle reads the project as it is (the metadata may be stale: a function
    may have been renamed, so some kinds may not apply)."""
    for k in rng.sample(FOLLOW_UPS, len(FOLLOW_UPS)):
        try:
            task = TASKS[k](rng, p)
        except (StopIteration, KeyError, ValueError):
            continue
        if task is not None:
            return task
    return None


def _maybe_small_talk(rng: random.Random, messages: list[dict]) -> None:
    if rng.random() < 0.05:
        _, prompt, answer = chat_turn(rng)
        messages += [{"role": "user", "content": noisy(rng, prompt, chat=True)}, {"role": "assistant", "content": answer}]


def turn_problem(rng: random.Random, conv: dict) -> dict:
    """A transcript cut before one of its trained assistant turns (a tool call, or the answer): the
    prompt of one turn of the agent, for distillation (the student writes the turn, its
    teacher grades every token)."""
    turns = [i for i, m in enumerate(conv["messages"]) if m["role"] == "assistant" and m.get("weight", 1) != 0]
    i = rng.choice(turns)
    return {"kind": "code", "messages": conv["messages"][:i], "tools": conv["tools"]}


# ---------------------------------------------------------------------------
# Pretraining: Python, before any chat
# ---------------------------------------------------------------------------

def _printed(v) -> str:
    return str(v)


def pretrain_document(rng: random.Random) -> str:
    """A document of the pretraining corpus: the files of a project, what its scripts print,
    functions with what they do in English, bugs and their fixes. No chat format and no tools:
    that is what SFT teaches."""
    p = make_project(rng)
    r = rng.random()
    if r < 0.45:
        # A repository dump, then a terminal session.
        parts = [f"# {path}\n{text}" for path, text in p.files.items()]
        session = []
        if p.main and p.calls:
            session.append(f"$ python3 {p.main}\n" + "".join(_printed(f.value(a)) + "\n" for f, a in p.calls))
        for t in p.tests:
            session.append(f"$ python3 {t}\nall tests passed\n")
        if session:
            parts.append("".join(session))
        return "\n".join(parts)
    if r < 0.75:
        # Documentation: what each function does, and examples.
        out = []
        for f in rng.sample(p.funcs(), len(p.funcs())):
            args = sample_args(rng, f.kind.args)
            ph = rng.choice(f.kind.phrases)
            if rng.random() < 0.5:
                out.append(f"`{f.sig}` {ph}.\n\n```python\n{f.code()}```\n\n>>> {f.call(args)}\n{lit(f.value(args))}\n")
            else:
                out.append(f"```python\n{f.code()}```\n\n`{f.name}` {ph}. For example, `{f.call(args)}` returns "
                           f"`{lit(f.value(args))}`.\n")
        return "\n".join(out)
    # Bugs and fixes.
    out = []
    for f in p.funcs():
        bug = body_bug(rng, f)
        if bug is None:
            continue
        bad_body, _ = bug
        bad, good = next((b, g) for b, g in zip(bad_body.split("\n"), f.kind.body.split("\n")) if b != g)
        out.append(f"Bug: `{f.name}` should {infinitive(rng.choice(f.kind.phrases))}, "
                   f"but it has `{bad.strip()}`.\n\n```python\n{f.code(bad_body)}```\n\nFix: replace `{bad.strip()}` "
                   f"with `{good.strip()}`.\n\n```python\n{f.code()}```\n")
    return "\n".join(out) or f"# {p.modules[0].path}\n{p.modules[0].text()}"


def infinitive(phrase: str) -> str:
    """"returns the sum of a and b" -> "return the sum of a and b"."""
    verb, _, rest = phrase.partition(" ")
    verb = verb[:-3] + "y" if verb.endswith("ies") else verb[:-1]
    return f"{verb} {rest}"


def text_of(conv: dict) -> list[str]:
    """The texts of a conversation, as the tokenizer will see them (for training it)."""
    out = []
    for m in conv["messages"]:
        out.append(m.get("content") or "")
        for call in m.get("tool_calls") or []:
            out += [f"{k}={v}" for k, v in json.loads(call["function"]["arguments"]).items()]
    return out


# ---------------------------------------------------------------------------
# Datasets: generated once, cached under data/code/
# ---------------------------------------------------------------------------

DATA_VERSION = 6   # bump when the world or the tasks change: cached sets are regenerated
CHUNK = 250


def data_dir() -> Path:
    return Path(os.environ.get("MINILAB_DATA_DIR", "data")) / "code"


def _chunk(job: tuple[int, int, dict]) -> list[dict]:
    seed, n, mix = job
    rng = random.Random(seed)
    kinds, weights = zip(*mix.items())
    return [conversation(rng, rng.choices(kinds, weights)[0]) for _ in range(n)]


def conversations(n: int, seed: int, mix: dict[str, float], workers: int | None = None) -> list[dict]:
    """n conversations drawn from `mix` ({kind: weight}), generated in parallel (every
    transcript runs its tools for real) and cached: the same arguments give the same set."""
    key_parts = [DATA_VERSION, n, seed, sorted(mix.items()), NAME]  # the model says its name
    key = hashlib.sha1(json.dumps(key_parts).encode()).hexdigest()[:10]
    path = data_dir() / f"conversations-{n}-{seed}-{key}.jsonl.gz"
    if path.exists():
        with gzip.open(path, "rt") as f:
            return [json.loads(line) for line in f]
    t0 = time.time()
    jobs = [(seed * 1_000_003 + i, min(CHUNK, n - i * CHUNK), mix) for i in range((n + CHUNK - 1) // CHUNK)]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        convs = [c for chunk in pool.imap(_chunk, jobs) for c in chunk]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with gzip.open(tmp, "wt") as f:
        for c in convs:
            f.write(json.dumps(c) + "\n")
    tmp.rename(path)
    print(f"generated {n} conversations in {time.time() - t0:.0f}s ({workers} workers) -> {path}", flush=True)
    return convs


def pretrain_documents(seed: int) -> Iterator[str]:
    """Endless stream of pretraining documents (cheap: nothing runs, outputs are computed)."""
    rng = random.Random(seed)
    while True:
        yield pretrain_document(rng)


def sft_set(cfg: dict, seed: int, size: int | None = None) -> list[dict]:
    """The config's agent transcripts ([sft] code_size and code_mix)."""
    sc = cfg["sft"]
    return conversations(size or sc["code_size"], seed, sc["code_mix"])


def main() -> None:
    """The data step of the speedrun: build (or find) the SFT set of agent transcripts."""
    import argparse
    import tomllib
    p = argparse.ArgumentParser(description="Generate the SFT set of agent transcripts (cached under data/code/).")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    convs = sft_set(cfg, cfg.get("seed", 0) + 2)
    print(f"SFT set: {len(convs)} agent transcripts")


if __name__ == "__main__":
    main()
