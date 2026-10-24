"""The coding agent: the sandbox's tools, the toy code world, and what the chat template does for agents."""

import json
import random

import pytest

from minilab.data import code, horizon
from minilab.data.loader import chat_batch, context_windows
from minilab.data.sandbox import OPENCODE_TOOLS, TOOL_SCHEMAS, Sandbox
from minilab.eval.horizon import DEFAULTS
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import (PromptTooLong, coerce_arguments, fit_messages, parse_completion, prepare_messages,
                                    render_conversation, render_prompt)


@pytest.fixture(scope="module")
def tok():
    rng = random.Random(0)
    convs = [code.conversation(rng, k) for k in list(code.TASKS) + ["chat", "title", "session"] for _ in range(3)]
    texts = [t for c in convs for t in code.text_of(c)] + [code.pretrain_document(rng) for _ in range(50)]
    return Tokenizer.train(texts, 700)


# ---- the sandbox: opencode 2.0.20's exact tool results ---------------------------------

def test_sandbox_tools_match_opencode(tmp_path):
    files = {"hello.py": 'def greet(name):\n    return "Hello, " + name\n\n\nprint(greet("world"))\n',
             "test_calc.py": "def add(a, b):\n    return a - b\n\n\nassert add(2, 3) == 5\n",
             "src/util.py": "x = 1\ny = x\n"}
    sb = Sandbox(files, root=tmp_path)
    root = str(sb.root)
    assert sb.call("read", {"path": "hello.py", "offset": 2, "limit": 2}) == (
        'Read file hello.py, lines 2-3\n2:     return "Hello, " + name\n3: \n'
        "[Output truncated. Continue reading with offset: 4]")
    assert sb.call("read", {"path": "src"}) == "Read directory src, entries 1-1\nutil.py"
    assert sb.call("read", {"path": "missing.py"}) == (
        '{"error":{"type":"tool.execution","message":"File not found: missing.py"},"content":[]}')
    assert sb.call("write", {"path": "new.py", "content": "A = 1\n"}) == "Created file successfully: new.py"
    assert sb.call("write", {"path": "new.py", "content": "A = 2\n"}) == "Wrote file successfully: new.py"
    assert sb.call("edit", {"path": "src/util.py", "oldString": "x", "newString": "z"}).startswith(
        '{"error":{"type":"tool.execution","message":"Found 2 matches for oldString, but expected exactly one.')
    assert sb.call("edit", {"path": "src/util.py", "oldString": "x", "newString": "z", "replaceAll": True}) == (
        "Edited src/util.py (2 replacements)")
    assert sb.call("glob", {"pattern": "*.js"}) == "No files found"
    assert sb.call("grep", {"pattern": "zzz"}) == "No matches found"
    grep = sb.call("grep", {"pattern": "def "})
    assert grep.startswith("Found 2 matches\n") and f"{root}/hello.py:\n  Line 1: def greet(name):\n" in grep
    out = sb.call("shell", {"command": "python3 test_calc.py"})
    assert out.startswith("Traceback") and out.endswith("AssertionError\n\nExited with code 1")
    assert sb.call("shell", {"command": "true"}) == "(no output)"
    assert "No tool named" in sb.call("bash", {"command": "ls"})


def test_sandbox_stays_in_the_project(tmp_path):
    sb = Sandbox({"a.py": "A = 1\n"}, root=tmp_path / "p")
    (tmp_path / "secret.txt").write_text("secret")
    for path in ["../secret.txt", str(tmp_path / "secret.txt"), "/etc/passwd"]:
        assert "Access denied" in sb.call("read", {"path": path})
        assert "Access denied" in sb.call("write", {"path": path, "content": "x"})
    (sb.root / "link").symlink_to(tmp_path / "secret.txt")
    assert "Access denied" in sb.call("read", {"path": "link"})


# ---- the toy world ---------------------------------------------------------------

@pytest.mark.parametrize("kind", list(code.EVAL_TASKS))
def test_oracles_solve_their_tasks(kind):
    """Every oracle transcript passes its task's check; doing nothing never does."""
    rng = random.Random(f"test-{kind}")
    for _ in range(3):
        task = code.make_task(rng, kind)
        with Sandbox(task.files) as sb:
            assert not task.check(sb, "")
            messages = code.play(task, sb)
            assert task.check(sb, messages[-1]["content"]), (task.prompt, messages)


def test_project_metadata_reads_back_from_files():
    """project_from_files finds what make_project made: modules, script, constant, calls, tests."""
    def meta(p):
        return (sorted((m.path, [(f.kind.key, f.name) for f in m.funcs]) for m in p.modules), p.main,
                {t: m.path for t, m in p.tests.items()}, p.const and (p.const.name, p.const.value, p.const.argtype),
                [(f.name, args) for f, args in p.calls])
    rng = random.Random(0)
    for _ in range(100):
        p = code.make_project(rng)
        assert meta(code.project_from_files(p.files)) == meta(p)


@pytest.mark.parametrize("family", horizon.FAMILIES)
def test_long_tasks_grow_with_k(family):
    """Long tasks are made solvable (their oracle passed every check) and get longer with k; the
    first request's check fails on the untouched project."""
    rng = random.Random(f"test-{family}")
    tasks = [horizon.make_long_task(rng, family, k) for k in (1, 3, 6)]
    assert [len(t.requests) for t in tasks] == ([1, 1, 1] if family == "repair" else [1, 3, 6])
    assert tasks[0].length < tasks[1].length < tasks[2].length
    for t in tasks:
        with Sandbox(t.files) as sb:
            first = t.requests[0]
            if first.kind not in code.READ_ONLY:
                assert not first.check(sb, "")


def test_repair_check_needs_real_fixes():
    """Editing a test to pass, or leaving one bug, fails the repair check."""
    rng = random.Random(1)
    task = next(t for t in iter(lambda: horizon.repair_task(rng, 2), "never") if t is not None)
    with Sandbox(task.files) as sb:
        messages = code.play(task, sb)
        assert task.check(sb, messages[-1]["content"])
    with Sandbox(task.files) as sb:
        for path in [p for p in task.files if p.startswith("test_")]:
            sb.write_file(path, 'print("all tests passed")\n')
        assert not task.check(sb, "Fixed.")


def test_training_never_starts_like_an_eval_task():
    """A conversation whose fingerprint is excluded (an eval task's) is redrawn."""
    first = code._chunk((7, 3, {"fix_test": 1.0}, frozenset()))
    again = code._chunk((7, 3, {"fix_test": 1.0}, frozenset({first[0]["fp"]})))
    assert len(again) == 3 and first[0]["fp"] not in {c["fp"] for c in again}
    assert code.fingerprint({"a.py": "x = 1\n"}, '"Please list the files."') == \
        code.fingerprint({"a.py": "x = 1\n"}, "list the files")


def test_benchmark_keeps_its_own_phrasings_and_lengths():
    """Training draws k from TRAIN_KS and never uses the benchmark's ways of asking."""
    assert not set(horizon.REPAIR_PROMPTS) & set(horizon.EVAL_REPAIR_PROMPTS)
    rng = random.Random(0)
    for family in ("repair", "session"):
        conv, _ = horizon.long_conversation(rng, family)
        users = [m["content"] for m in conv["messages"] if m["role"] == "user"]
        assert not any(p.lower().rstrip(".") in u.lower() for u in users for p in horizon.EVAL_REPAIR_PROMPTS)
        assert not any(u.strip('"').startswith(tuple(horizon.EVAL_LEAD_INS)) for u in users)
    lo, hi = horizon.TRAIN_KS["session"]
    assert max(DEFAULTS["ks"]["session"]) > hi and max(DEFAULTS["ks"]["repair"]) > horizon.TRAIN_KS["repair"][1]


def test_context_windows_are_what_the_server_sends(tok):
    """Each window of a long transcript is the budgeted prompt before one of its turns, then that
    turn, the only tokens trained on."""
    conv = code.conversation(random.Random(4), "repair")
    rows = context_windows(tok, conv, 300, 50, random.Random(0))
    msgs = conv["messages"]
    turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("weight", 1) != 0]
    assert len(rows) == len(turns)
    for i, row in zip(turns, rows):
        prompt = render_prompt(tok, msgs[:i], conv["tools"], budget=300)
        assert list(row["ids"][:row["context"]]) == prompt and len(prompt) <= 300
    x, y = chat_batch(tok, rows, 1024)
    assert all((yy[:r["context"] - 1] == -1).all() and (yy[r["context"] - 1:len(r["ids"]) - 1] >= 0).all()
               for yy, r in zip(y, rows))


def test_slipped_edits_never_go_through():
    """The edits shown as mistakes must fail: a slip that is a piece of the line would edit it."""
    rng = random.Random(0)
    for old in ["    return a - b", "    if x > 0:", 'print(greet("Ada"))', "    total = 0"]:
        assert all(code._slip(rng, old) not in old for _ in range(200))


def test_conversations_are_deterministic_and_relocated():
    a = code.conversation(random.Random(3), "fix_test")
    b = code.conversation(random.Random(3), "fix_test")
    assert a == b
    text = json.dumps(a)
    assert "minicode-" not in text  # the sandbox's temporary directory is gone
    assert a["tools"] == OPENCODE_TOOLS or set(OPENCODE_TOOLS) <= set(a["tools"])


# ---- the chat template, for agents ---------------------------------------------------

def test_tool_calls_round_trip(tok):
    call = {"id": "c1", "type": "function", "function": {"name": "edit", "arguments": json.dumps(
        {"path": "calc.py", "oldString": '    return "a" - b', "newString": "    return a + b\n", "replaceAll": True})}}
    ids, mask = render_conversation(tok, [{"role": "user", "content": "fix"},
                                          {"role": "assistant", "content": "", "tool_calls": [call]}])
    start = ids.index(tok.special("<|assistant_start|>")) + 1
    assert all(mask[start:]) and not any(mask[:start])
    parsed = parse_completion(tok, ids[start:])
    assert parsed.finished and parsed.content == ""
    (c,) = parsed.tool_calls
    args = json.loads(coerce_arguments(c["arguments"], TOOL_SCHEMAS["edit"]))
    assert c["name"] == "edit" and args == json.loads(call["function"]["arguments"])
    assert '\\"' not in tok.decode(ids)  # code goes in raw: no JSON escaping


def test_prepare_messages_trims_what_the_model_cant_use():
    system = ("You are an AI agent running in OpenCode, a coding agent harness. Help the user.\n" + "rules " * 5000
              + "\n<env>\n  Working directory: /Users/me/proj\n</env>")
    out = prepare_messages([{"role": "system", "content": system},
                         {"role": "user", "content": "Read /Users/me/proj/calc.py"},
                         {"role": "tool", "content": "/Users/me/proj/src/a.py\n/Users/me/proj/b.py"},
                         {"role": "tool", "content": '{"error":{"type":"tool.execution","message":"File not found: x.py"},'
                                                     '"content":[]}'}])
    assert out[0]["content"] == "You are an AI agent running in OpenCode, a coding agent harness."
    assert out[1]["content"] == "Read calc.py"
    assert out[2]["content"] == "src/a.py\nb.py"
    assert out[3]["content"] == "Error: File not found: x.py"


def test_fit_messages_keeps_the_current_request(tok):
    rng = random.Random(1)
    convs = [code.conversation(rng, "fix_test") for _ in range(4)]
    system = convs[0]["messages"][0]
    history = [m for c in convs for m in c["messages"][1:]]   # four requests in a row
    full = len(render_prompt(tok, [system, *history], OPENCODE_TOOLS))
    budget = full // 2
    ids = render_prompt(tok, [system, *history], OPENCODE_TOOLS, budget=budget)
    assert len(ids) <= budget < full
    last_user = [m for m in history if m["role"] == "user"][-1]["content"]
    assert tok.encode(last_user)[:5] == ids[ids.index(tok.special("<|user_start|>")) + 1:][:5] or \
        last_user in tok.decode(ids)
    kept = fit_messages(tok, prepare_messages([system, *history]), OPENCODE_TOOLS, budget)
    assert kept[0]["role"] == "system" and kept[1]["role"] == "user"   # whole turns only
    with pytest.raises(PromptTooLong):
        render_prompt(tok, [system, {"role": "user", "content": "word " * 2000}], OPENCODE_TOOLS, budget=200)


def test_coerce_arguments():
    schema = {"properties": {"offset": {"type": "integer"}, "replaceAll": {"type": "boolean"},
                             "content": {"type": "string"}}}
    args = json.loads(coerce_arguments(json.dumps({"offset": "5", "replaceAll": "true", "content": "42"}), schema))
    assert args == {"offset": 5, "replaceAll": True, "content": "42"}
    assert coerce_arguments('{"offset": "five"}', schema) == '{"offset": "five"}'


def test_tokenizer_save_and_load(tok, tmp_path):
    tok.save(tmp_path / "tok.json")
    loaded = Tokenizer.load(tmp_path / "tok.json")
    assert loaded.special_tokens == tok.special_tokens and "<|arg|>" in loaded.special_tokens
    text = code.pretrain_document(random.Random(5))
    assert loaded.encode(text) == tok.encode(text)
