# mini-lab in opencode

opencode 2 driving `mini-4` (the one model that also tells stories and adds numbers) or
`mini-code-1` (the coding specialist), everything in containers: the mini-lab inference server
and API (serving only those two), and opencode with a small demo project whose `add` returns
`a - b`. See [docs/opencode.md](../../docs/opencode.md) for how the coding agent was made, and
[docs/mini-4.md](../../docs/mini-4.md) for the single model.

```bash
bash speedrun.sh unified                                 # trains and releases models/mini-4 (~1h15 on MPS)
bash examples/opencode/demo.sh                           # opencode's TUI: you approve each action
bash examples/opencode/demo.sh run --auto "Run the tests and fix any bug"
bash examples/opencode/demo.sh run --auto -m minilab/mini-code-1 "Run the tests and fix any bug"
```

Needs Docker. Try "What files are in this project?", "What does main.py print?", "Rename
multiply to times", "Add a function square to calc.py that returns x squared", "Change NAME to
Bob and run main.py", "Create hello.py that prints Hello, world! and run it". One request at a
time works best: it's a 5M-parameter model.

**Nothing runs on your machine.** The containers share an internal Docker network: opencode,
and every command the model runs through its `shell` tool, can reach the mini-lab API and
nothing else, not even the internet. Each `demo.sh` starts from a fresh copy of the project.

| file | |
|---|---|
| `Dockerfile` | opencode 2.0.20 (checked against its sha256), Python, the demo project, and `opencode.json` as the user's config |
| `opencode.json` | the mini-lab provider: `@ai-sdk/openai-compatible` at `http://api:8000/v1`, mini-4 and mini-code-1 with a large context so opencode never compacts (the server fits requests itself) |
| `docker-compose.yml` | inference, API (which creates a local account and hands its key to opencode) and opencode, on the internal network |
| `project/` | the demo project |
