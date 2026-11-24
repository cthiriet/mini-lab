"""At T=1 (RL's rollouts), per digit count: how often a checkpoint answers directly, and how right each way is."""
import random, sys
sys.path.insert(0, "/Users/clementthiriet/VM/mini-lab/.claude/worktrees/agent-ae2f7c19e5dfffeb8")
from minilab.checkpoint import load_checkpoint
from minilab.eval.tasks import arithmetic_problems, generate, grade
from minilab.tokenizer.chat import reasoning_length, render_prompt
for run in sys.argv[1:]:
    model, tok, _ = load_checkpoint(run, device="cpu")
    stop = {tok.special("<|assistant_end|>")}
    print(run)
    for d in [1, 2, 3, 4, 5]:
        ps = arithmetic_problems(d, 200)
        outs = generate(model, [render_prompt(tok, p["messages"], p["tools"]) for p in ps], 200, stop, temperature=1.0, seed=7)
        th = [(reasoning_length(tok, o) > 0, grade(tok, o, p)) for o, p in zip(outs, ps)]
        direct = [ok for t, ok in th if not t]; think = [ok for t, ok in th if t]
        f = lambda v: f"{100*sum(v)/len(v):.0f}%" if v else "-"
        print(f"  {d}d direct {100*len(direct)/len(th):.0f}% (right {f(direct)}), thinking right {f(think)}")
