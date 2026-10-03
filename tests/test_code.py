"""mini-code: the sandbox's tools, the toy code world, and the "code" chat template."""

import json
import random

import pytest

from minilab.data import code
from minilab.data.sandbox import OPENCODE_TOOLS, TOOL_SCHEMAS, Sandbox
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import (PromptTooLong, code_messages, coerce_arguments, fit_messages, parse_completion,
                                    render_conversation, render_prompt)


@pytest.fixture(scope="module")
def tok():
    rng = random.Random(0)
    convs = [code.conversation(rng, k) for k in list(code.TASKS) + ["chat", "title", "session"] for _ in range(3)]
    texts = [t for c in convs for t in code.text_of(c)] + [code.pretrain_document(rng) for _ in range(50)]
    return Tokenizer.train(texts, 700, chat_template="code")


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

@pytest.mark.parametrize("kind", list(code.TASKS))
def test_oracles_solve_their_tasks(kind):
    """Every oracle transcript passes its task's check and its reward's guard; doing nothing never does."""
    rng = random.Random(f"test-{kind}")
    for _ in range(3):
        task = code.make_task(rng, kind)
        with Sandbox(task.files) as sb:
            assert not task.check(sb, "") and not task.reward(sb, "", [])
            messages = code.play(task, sb)
            assert task.check(sb, messages[-1]["content"]), (task.prompt, messages)
            assert task.reward(sb, messages[-1]["content"], code.calls_of(messages)), (task.prompt, messages)


def _task_where(kind: str, ok, tries: int = 200) -> code.Task:
    rng = random.Random(f"hack-{kind}")
    for _ in range(tries):
        task = code.make_task(rng, kind)
        if ok(task):
            return task
    raise AssertionError(kind)


def test_guards_catch_what_the_checks_miss():
    """Ways to pass a check without doing the task, that an RL reward must not pay for."""
    # fix_test: the failing assert deleted instead of the bug fixed
    task = _task_where("fix_test", lambda t: True)
    with Sandbox(task.files) as sb:
        for path in [p for p in task.files if p.startswith("test_")]:
            sb.write(path, "\n".join(l for l in task.files[path].splitlines() if not l.startswith("assert")) + "\n")
        assert task.check(sb, "Fixed.") and not task.reward(sb, "Fixed.", [])
    # run_tests: "All tests pass." without running them
    task = _task_where("run_tests", lambda t: True)
    with Sandbox(task.files) as sb:
        answer = "All tests pass. test passes."
        if task.check(sb, answer):
            assert not task.reward(sb, answer, [("glob", {"pattern": "**/test_*.py"})])
    # find_def: every line number at once
    task = _task_where("find_def", lambda t: "couldn't" not in t.title)
    with Sandbox(task.files) as sb:
        answer = " ".join(f"{p}, line {n}." for p in task.files for n in range(1, 30))
        assert task.check(sb, answer) and not task.reward(sb, answer, [("grep", {"pattern": "def"})])


def test_conversations_are_deterministic_and_relocated():
    a = code.conversation(random.Random(3), "fix_test")
    b = code.conversation(random.Random(3), "fix_test")
    assert a == b
    text = json.dumps(a)
    assert "minicode-" not in text  # the sandbox's temporary directory is gone
    assert a["tools"] == OPENCODE_TOOLS or set(OPENCODE_TOOLS) <= set(a["tools"])


# ---- the "code" chat template ------------------------------------------------------

def test_code_template_round_trip(tok):
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


def test_code_template_trims_what_the_model_cant_use():
    system = ("You are an AI agent running in OpenCode, a coding agent harness. Help the user.\n" + "rules " * 5000
              + "\n<env>\n  Working directory: /Users/me/proj\n</env>")
    out = code_messages([{"role": "system", "content": system},
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
    kept = fit_messages(tok, code_messages([system, *history]), OPENCODE_TOOLS, budget)
    assert kept[0]["role"] == "system" and kept[1]["role"] == "user"   # whole turns only
    with pytest.raises(PromptTooLong):
        render_prompt(tok, [system, {"role": "user", "content": "word " * 2000}], OPENCODE_TOOLS, budget=200)


def test_coerce_arguments():
    schema = {"properties": {"offset": {"type": "integer"}, "replaceAll": {"type": "boolean"},
                             "content": {"type": "string"}}}
    args = json.loads(coerce_arguments(json.dumps({"offset": "5", "replaceAll": "true", "content": "42"}), schema))
    assert args == {"offset": 5, "replaceAll": True, "content": "42"}
    assert coerce_arguments('{"offset": "five"}', schema) == '{"offset": "five"}'


def test_tokenizer_keeps_its_template(tok, tmp_path):
    tok.save(tmp_path / "tok.json")
    loaded = Tokenizer.load(tmp_path / "tok.json")
    assert loaded.chat_template == "code" and "<|arg|>" in loaded.special_tokens
    default = Tokenizer.train(["hello world"] * 10, 300)
    default.save(tmp_path / "default.json")
    assert Tokenizer.load(tmp_path / "default.json").chat_template == "default"
