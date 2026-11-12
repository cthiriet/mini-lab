"""The calculator in the reasoning: think_result, GPT.generate(force=...), and the engine."""

import asyncio

import torch

from minilab.data.arithmetic import think_result
from minilab.inference.calculator import ThinkCalculator, think_force
from minilab.model.gpt import GPT, GPTConfig
from tests.test_inference_engine import ScriptedModel, prompt_for, release, make_runner, run  # noqa: F401


def test_think_result():
    assert think_result("70 - 32 =") == " 38"
    assert think_result("70-32=") == "38"
    assert think_result("so (19 + 20 - 31) x 2 =") == " 16"
    assert think_result("3 + 4 x 2 =") == " 11"
    assert think_result("12 - 3 = 9, 9 + 1 =") == " 10"
    assert think_result("7 + a =") is None
    assert think_result("7 + =") is None
    assert think_result("70 - 80 =") is None   # negative: left to the model
    assert think_result("38 =") is None        # no operation
    assert think_result("(38) =") is None
    assert think_result("3 4 + 2 =") is None
    assert think_result("70 - 32") is None     # no equal sign


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=300, block_size=64, n_layer=2, n_head=2, n_embd=32)).eval()


def test_generate_force():
    model, prompts = tiny(), [[1, 2, 3], [4, 5]]
    plain = model.generate(prompts, 12, temperature=0)
    assert model.generate(prompts, 12, temperature=0, force=None) == plain
    assert model.generate(prompts, 12, temperature=0, force=lambda s, o, t: []) == plain
    # after the 3rd token of slot 1, write 7, 8, 9; then it goes on from there
    forced = model.generate(prompts, 12, temperature=0, force=lambda s, o, t: [7, 8, 9] if s == 1 and len(o) == 3 else [])
    assert forced[0] == plain[0] and forced[1][:6] == plain[1][:3] + [7, 8, 9]
    prefix = prompts[1] + forced[1][:6]
    assert forced[1][6:] == model.generate([prefix], 6, temperature=0)[0]  # the forced tokens went through the cache


def test_calculator_state(release):
    _, _, tok = release
    S, enc = tok.special, tok.encode

    def feed(c, text):  # the last token's answer (the test tokenizer has no merges: " =" is two tokens)
        return [c.feed(t) for t in enc(text)][-1]

    c = ThinkCalculator(tok)
    assert feed(c, "70 - 32 =") == []                                   # not thinking
    assert c.feed(S("<|think_start|>")) == []
    assert feed(c, "Step 1: 70 - 32") == []
    assert feed(c, " =") == enc(" 38")
    assert feed(c, " 38. And 5+2=") == enc("7")
    c.feed(S("<|think_end|>"))
    assert feed(c, "70 - 32 =") == []                                   # the reasoning is over
    force = think_force(tok, [[1], [2, S("<|think_start|>")]])          # a prompt that opens the reasoning
    assert [force(1, [], t) for t in enc("9 - 4 =")][-1] == enc(" 5")
    assert [force(0, [], t) for t in enc("9 - 4 =")][-1] == []


def test_engine_calculator(release, make_runner):  # noqa: F811
    _, model, tok = release
    S = tok.special
    # The scripted model writes " 9" after "7 - 3 =": with the calculator, " 4" is written instead (same length,
    # so the script goes on in step) and counted and streamed like the other reasoning tokens.
    script = [S("<|think_start|>"), *tok.encode("7 - 3 = 9 ok"), S("<|think_end|>"), *tok.encode("4"), S("<|assistant_end|>")]

    async def main(**params):
        runner = make_runner(model=ScriptedModel(model.config, script))
        return await run(runner, prompt_for(tok, "What is 7 - 3?"), temperature=0, **params)

    req, _, done = asyncio.run(main())
    assert done["reasoning"] == "7 - 3 = 4 ok" and done["content"] == "4"
    assert done["usage"]["completion_tokens_details"]["reasoning_tokens"] == len(tok.encode("7 - 3 = 4 ok")) + 2
    assert done["usage"]["completion_tokens"] == len(script)
    _, _, off = asyncio.run(main(think_calculator=False))
    assert off["reasoning"] == "7 - 3 = 9 ok"
    _, _, none = asyncio.run(main(think=False, max_tokens=3))  # think=False still bans <|think_start|>
    assert none["reasoning"] is None
