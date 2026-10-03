"""The recipe: one model for stories, addition and code."""

import random

import pytest

from minilab.data import arithmetic, code
from minilab.data.conversations import NAME
from minilab.data.loader import chat_batch, mixture, pretrain_documents
from minilab.eval import gate
from minilab.eval.tasks import grade, turn_ok
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import parse_completion, render_conversation


@pytest.fixture(scope="module")
def tok():
    rng = random.Random(0)
    texts = ["Once upon a time, there was a little dog named Tom."] * 20
    texts += [arithmetic.pretrain_document(rng, [1, 2, 3]) for _ in range(30)]
    texts += [t for k in ("fix_test", "chat") for _ in range(3) for t in code.text_of(code.conversation(rng, k))]
    return Tokenizer.train(texts + ["calculator", "expression=1 + 2"], 700)


def test_calculator_calls(tok):
    """The calculator's calls are written like opencode's (raw arguments), and graded as before."""
    messages = [{"role": "user", "content": "What is 347 + 58?"}, *arithmetic.tool_round_trip(347, 58)[:1]]
    ids, mask = render_conversation(tok, messages, ["calculator"])
    start = ids.index(tok.special("<|assistant_start|>")) + 1
    completion = ids[start:]
    assert tok.special("<|arg|>") in completion and "expression=347 + 58" in tok.decode(completion)
    problem = {"kind": "tool", "messages": messages[:1], "tools": ["calculator"], "a": 347, "b": 58, "answer": 405}
    assert grade(tok, completion, problem) and turn_ok(tok, completion, True)
    (call,) = parse_completion(tok, completion).tool_calls
    assert arithmetic.call_expression(call) == "347 + 58"
    # an <|arg|> outside a call is still a stray special token
    answer = tok.encode("The answer is 405.") + [tok.special("<|arg|>"), tok.special("<|assistant_end|>")]
    assert not grade(tok, answer, {**problem, "kind": "add"})


def test_the_agent_says_its_name():
    rng = random.Random(1)
    answers = [code.chat_turn(rng) for _ in range(40)]
    assert NAME == "prelude" and any(a.startswith("I'm prelude, ") for k, _, a in answers if k == "identity")


def test_turn_problems_end_before_a_trained_turn():
    rng = random.Random(2)
    conv = code.conversation(rng, "fix_test")
    for _ in range(10):
        p = code.turn_problem(rng, conv)
        nxt = conv["messages"][len(p["messages"])]
        assert p["kind"] == "code" and nxt["role"] == "assistant" and nxt.get("weight", 1) != 0
        assert p["messages"][-1]["role"] in ("user", "tool") and p["tools"] == conv["tools"]


def test_pretraining_mixes_python_in(tok):
    stories = [f"Once upon a time, there was a dog number {i}." for i in range(50)]
    docs = pretrain_documents(tok, stories, [1, 2], 0.2, seed=0, code_frac=0.5)
    texts = [tok.decode(next(docs)[1:]) for _ in range(200)]
    assert any("def " in t for t in texts) and any(t.startswith("Once upon") for t in texts)
    plain = pretrain_documents(tok, stories, [1, 2], 0.2, seed=0)
    assert not any("def " in tok.decode(next(plain)) for _ in range(200))
    mixed = mixture([iter(lambda: "a", None), iter(lambda: "b", None)], [0.9, 0.1], seed=0)
    draws = [next(mixed) for _ in range(1000)]
    assert 50 < draws.count("b") < 150


def test_sft_rows_documents_and_padding(tok):
    """SFT replays pretraining documents (trained on every token); rows are padded to a
    multiple of 64 (on MPS, every new shape keeps new kernels)."""
    doc = [tok.bos_id, *tok.encode("Once upon a time, there was a dog.")]
    conv = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}], "tools": None}
    x, y = chat_batch(tok, [{"ids": doc}, conv], 1024)
    assert x.shape[1] == 64 and (y[0] != -1).sum() == len(doc) - 1
    assert 0 < (y[1] != -1).sum() < len(tok.encode("Hello!")) + 2


def test_the_platform_offers_only_the_served_models(tmp_path, monkeypatch):
    """A deployment that keeps old releases on disk but serves one ($MINILAB_SERVE_MODELS)."""
    from types import SimpleNamespace

    from minilab.platform.web import released_models
    from minilab.registry import ModelInfo, Pricing, write_release
    from minilab.settings import get_settings
    for i, model_id in enumerate(["prelude-1", "prelude-2", "prelude-3"]):
        (tmp_path / model_id).mkdir()
        write_release(tmp_path / model_id, ModelInfo(id=model_id, created=i, description="", context_length=256,
                                                     pricing=Pricing(1, 1)))
    monkeypatch.setenv("MINILAB_MODELS_DIR", str(tmp_path))

    def offered(serve: str) -> list[str]:
        monkeypatch.setenv("MINILAB_SERVE_MODELS", serve)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=get_settings())))
        return [m.id for m in released_models(request)]

    assert offered("") == ["prelude-3", "prelude-2", "prelude-1"]   # newest first
    assert offered("prelude-2") == ["prelude-2"]


def test_the_gate_reads_both_evals():
    """The gate reads the main eval's metrics and the coding eval's, whose clashing names get a prefix."""
    cfg = {"data": {"digits": [1]}, "eval": {"n_per_digit": 10, "n_sampled": 10, "n_tool": 10, "n_chat": 10},
           "code_eval": {"n_per_kind": 10, "n_chat": 10}}
    result = {"val_ppl": 5.0, "val_bpc": 1.1, "arithmetic": {"1": 1.0}, "chat": 0.8,
              "code": {"val_ppl": 1.3, "val_bpc": 0.2, "tasks": {"fix_test": 0.9}, "chat": 1.0,
                       "title": 1.0, "valid_calls": 1.0}}
    m = gate.metrics(result, cfg)
    assert m["bpc"][0] == 1.1 and m["code bpc"][0] == 0.2 and m["chat"][0] == 0.8 and m["code chat"][0] == 1.0
    assert m["fix_test"][0] == 0.9 and "ppl" not in m
    worse = {**result, "code": {**result["code"], "tasks": {"fix_test": 0.5}}}
    assert [c.metric for c in gate.compare(worse, result, cfg) if c.regressed] == ["fix_test"]
