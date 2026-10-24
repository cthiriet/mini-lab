"""The microscope (interp/trace.py) computes exactly what the model computes, and its parts add up."""

import pytest
import torch

from minilab.interp.trace import components, logit_lens, trace
from minilab.model.gpt import GPT, GPTConfig


@pytest.fixture(params=[False, True], ids=["plain", "prelude-block"])
def model(request):
    torch.manual_seed(0)
    on = request.param
    cfg = GPTConfig(vocab_size=50, block_size=32, n_layer=3, n_head=2, n_embd=16, mlp="relu2" if on else "gelu",
                    value_residual=on, qk_norm=on, x0_mix=on, softcap=15.0 if on else 0.0)
    m = GPT(cfg).eval()
    if on:  # trained blocks don't keep the initial mixes
        with torch.no_grad():
            for b in m.blocks:
                b.x0.copy_(torch.tensor([0.8, 0.3]))
                if b.attn.value_mix is not None:
                    b.attn.value_mix.fill_(0.3)
    return m


IDS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7]


def test_trace_matches_the_model(model):
    tr = trace(model, IDS)
    logits, _ = model(torch.tensor([IDS]))
    assert torch.allclose(tr.logits, logits[0], atol=1e-5)
    assert torch.allclose(logit_lens(model, tr)[-1], tr.logits)
    assert all(torch.allclose(p.sum(-1), torch.ones(p.shape[:2])) for p in tr.pattern)   # each row a distribution
    assert all((p.triu(1) == 0).all() for p in tr.pattern)                                 # causal


def test_components_add_up_to_the_residual_stream(model):
    tr = trace(model, IDS)
    assert torch.allclose(sum(components(tr).values()), tr.resid[-1], atol=1e-5)


def test_patches_change_what_they_touch(model):
    same = trace(model, IDS, {("mlp", 1): lambda x: x})
    assert torch.equal(same.logits, trace(model, IDS).logits)

    def kill(x):
        x = x.clone()
        x[0] = 0
        return x

    ablated = trace(model, IDS, {("heads", 2): kill})
    assert (ablated.heads[2][0] == 0).all() and not torch.allclose(ablated.logits, same.logits)
    assert torch.equal(ablated.resid[2], same.resid[2])   # what comes before the patch is untouched
