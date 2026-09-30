# mini-code: a coding agent for opencode

mini-code is a second model family, next to mini: a ~5M-parameter GPT trained from scratch on
a laptop to be the model behind [opencode](https://opencode.ai), the terminal coding agent.
It reads requests like "Run the tests and fix any bug", calls opencode's tools (`read`,
`write`, `edit`, `glob`, `grep`, `shell`) until the job is done, and says what it did.

It is a demonstration, not a coder: its whole world is small Python projects of a few files
and little functions (`add`, `greet`, `reverse`...). The point is that tool use, the format a
real agent harness speaks and a loop of calls and results can be learned by a model this
small, in minutes, and served through the same OpenAI-compatible API as any other model.

```bash
bash speedrun.sh code                         # data, tokenizer, pretraining, SFT, eval -> models/mini-code-1
bash examples/opencode/demo.sh                # opencode's TUI on a demo project, in a container
bash examples/opencode/demo.sh run --auto "Run the tests and fix any bug"
```

## Results

`runs/code`, released as `models/mini-code-1`: `bash speedrun.sh code` takes 24 minutes on an
Apple M5 Pro (3 min of data, 8 min of pretraining and 12 min of SFT on MPS, 1 min of eval). Every
column but `ppl` is a task played end to end, greedy, with new projects:

```
stage              ppl  agent  explore  create  modify  repair  valid calls  chat  title  train time
---------------  -----  -----  -------  ------  ------  ------  -----------  ----  -----  ----------
pretrain (base)   1.25      -        -       -       -       -            -     -      -     8.2 min
sft              23.15    98%      99%    100%     96%     97%         100%  100%   100%    11.9 min
```

| task (50 each) | success | | task (50 each) | success |
|---|---:|---|---|---:|
| list_files | 98% | | create_script | 100% |
| find_def | 100% | | rename | 100% |
| show_file | 100% | | change_const | 96% |
| explain | 100% | | add_func | 92% |
| run | 98% | | fix_test | 100% |
| run_tests | 100% | | fix_crash | 94% |
| create_func | 100% | | | |

`ppl` is the perplexity on held-out Python documents: SFT raises it, as the model now writes
transcripts instead of documents. The base model has no chat format, so it can't be asked to do
anything. The failures left are mostly copies: a misspelled name to copy from a traceback
(`backwaards`), a whole file to write back with one function more.

And in opencode itself (`bash examples/opencode/demo.sh run --auto "Run the tests and fix any bug"`,
on the demo project whose `add` returns `a - b`):

```
✱ Glob "**/test_*.py" 1 match
$ python3 test_calc.py
Traceback (most recent call last):
  File "/home/dev/project/test_calc.py", line 3, in <module>
    assert add(2, 3) == 5
AssertionError
✱ Grep "def add" 1 match
→ Read calc.py
← Edit calc.py
-    return a - b
+    return a + b
$ python3 test_calc.py
all tests passed
Fixed `add` in `calc.py`: it used `return a - b` instead of `return a + b`. The tests pass now.
```

Each of these requests reaches the API as ~35,000 characters; the model sees 20 to 370 tokens
of it, and answers in 60 to 300 ms.

### What the demo taught the data

The first model scored 98% on the eval and still failed on the demo project. Each failure
was a shortcut the synthetic data allowed, so each fix is in `data/code.py`:

- **It fixed the wrong line.** The demo's `multiply` returns `a * b`, which is exactly one
  of `add`'s bugs in the training data: the model had learned to find "a line that looks
  like a bug", not the line of the failing function. Now half the broken modules have such
  a distractor.
- **It copied from memory.** Asked to add a function, it rewrote the file with `add` fixed:
  every file it had ever copied was written the canonical way. Now a third of the files it
  adds to have a function that isn't, and the copy must keep it.
- **It gave up after a bad edit.** Now 15% of the fixes show a first edit whose `oldString`
  doesn't match, then the file read again and the right edit. The mistake is in the
  transcript with `"weight": 0` (as in OpenAI's fine-tuning format): the model sees it,
  but isn't trained to make it. `fix_crash` went from 77% to 96%.
- **An unknown file broke it.** opencode's own `opencode.json` in the listing: never seen,
  so the model started a `write` call it never closed. Now projects have READMEs,
  `opencode.json`, `requirements.txt`...
- **It repeated its last answer.** "Run main.py" after a rename got "Renamed `main.py`.", the
  same failure as mini's multi-turn chats. Now sessions also chain a change and more requests
  on the changed project.

## What opencode sends

Captured from opencode 2.0.20 with a fake OpenAI server (the model is declared with
`@ai-sdk/openai-compatible`, see `examples/opencode/opencode.json`):

- `POST /v1/chat/completions` with `stream: true`, `stream_options.include_usage`,
  `max_completion_tokens` = the model's `limit.output`, no temperature, and unknown extras
  (`store`, `reasoning_effort`) that the API ignores.
- A system prompt of ~20,000 characters (rules, environment, skills) and 12 tools with
  ~14,000 characters of JSON schemas: `edit, glob, grep, question, read, shell, skill,
  subagent, webfetch, websearch, write, execute`. That is ~10,000 tokens before the user's
  first word, for a model with a 1,024-token context.
- A separate request for the session's title: no tools, a "You are a title generator"
  system prompt, and the user's message.
- Tool results as opencode formats them, e.g. `read`:

  ```
  Read file calc.py, lines 1-2
  1: def add(a, b):
  2:     return a - b
  ```

  `glob` and `grep` list absolute paths, `shell` adds `Exited with code 1` when a command
  fails, and a failed call comes back as JSON: `{"error": {"type": "tool.execution",
  "message": "File not found: x.py"}, "content": []}`.

opencode 1.x differs (`bash` instead of `shell`, `filePath` instead of `path`, other result
formats): mini-code is trained on opencode 2's.

## Fitting an agent into 1,024 tokens

Small open models do the same: the chat template decides what the model sees. mini-code's
"code" template (`minilab/tokenizer/chat.py`, chosen by its tokenizer):

- **Tools by name only.** The model learned opencode's tools in its weights; the line
  `tools: edit, glob, ...` tells it which are there, the JSON schemas are dropped (as for
  mini's calculator).
- **The system prompt's first sentence**: "You are an AI agent running in OpenCode, a coding
  agent harness." A 5M model can't read 5,000 tokens of rules; it was trained on the
  behavior instead. The title generator's prompt becomes "You are a title generator."
- **Paths relative to the working directory**, taken from the system prompt's `Working
  directory:` line: `/Users/me/project/calc.py` becomes `calc.py`. Tool errors in opencode's
  JSON become `Error: <message>`.
- **Raw arguments.** A tool call is `<|tool_call_start|>edit<|arg|>path=calc.py<|arg|>oldString=    return a - b<|arg|>newString=    return a + b<|tool_call_end|>`:
  code goes in as it is, with its quotes and newlines, where JSON would need `\"` and `\n`
  (Qwen3-Coder moved from JSON to XML-style tool calls for the same reason). The API turns it
  into ordinary `tool_calls`, and types the values with the request's JSON schemas
  (`"true"` becomes `true` where the schema says boolean).
- **Server-side context management.** opencode compacts a session when it nears the model's
  context, by asking the model for a summary, which mini-code can't write. So its release
  says `"truncation": "auto"`: the inference server keeps the prompt within the context minus
  the room for the answer, by dropping the oldest turns, then cutting the current request's
  tool results. `opencode.json` declares a large context so opencode never compacts.
- **Greedy by default.** opencode sends no temperature, and the API's default is 1, too hot
  for a tiny model copying code: the release sets `"default_temperature": 0`, like a Hugging
  Face `generation_config.json`.

## The toy code world

`minilab/data/code.py`. A project is a few files: modules of 1-3 functions drawn from 42
kinds (`add`, `is_even`, `greet`, `reverse`, `total`, `factorial`...) under varied names, a
script that prints some calls (maybe through a constant like `NAME = "Ada"`), a test file of
asserts, a README. A task is a request about a project with an oracle that solves it with
the tools and a check that tells whether an attempt did:

| family | tasks |
|---|---|
| explore | list the files, find where a function is defined, show what a file defines, explain a function, run a script, run the tests |
| create | a file with a function from its description ("returns the sum of a and b"), a script that prints something |
| modify | rename a function in every file that uses it, change a constant, add a function to a file |
| repair | a failing test (run it, find the function, read it, fix the line, run it again), a script crashing on a typo |

Plus two requests in a row, small talk, "Who are you?", out-of-scope requests (politely
declined) and opencode's title requests.

**Transcripts are played for real.** The oracle's tool calls run in a sandbox
(`minilab/data/sandbox.py`) that reimplements opencode's tools with their exact output
formats, and `shell` really runs Python: the tracebacks, `Exited with code 1` and test
outputs in the training data are the real thing. 80,000 transcripts take under 3 minutes on 16
processes, and are cached in `data/code/`.

## Training

Two stages on the same pipeline as mini (`configs/code.toml`):

1. **Pretraining** on Python: the files of random projects with what their scripts print,
   functions with what they do in English, and bugs with their fixes. No chat and no tools.
2. **SFT** on the agent transcripts, loss on the assistant's turns only (tool calls and
   answers).

## Eval, in a sandbox

`minilab/eval/code.py` plays new tasks end to end, like opencode would: the model's messages,
its tool calls, their results, until it answers. The model's tool calls never run on the host:
every task gets a directory in one Docker container started with `--network none`,
`--read-only`, `--cap-drop ALL`, `--user nobody` and memory, CPU and process limits, and the
sandbox refuses paths outside the project. Each task is then checked on the project's files
and the answer: the tests pass, the new function returns the right values, the answer
contains what the script printed...

## Trying it in opencode

`examples/opencode/` runs everything in containers: the inference server and API (serving
only mini-code-1), and opencode 2.0.20 with a demo project that has a bug. Their network is
internal: opencode, and whatever the model runs through its `shell` tool, can reach the
mini-lab API and nothing else. `demo.sh` starts it with a fresh internal token; the API
container creates a local account with credits and hands its key to opencode.

To use mini-code with your own opencode instead, serve it (`./scripts/serve.sh`), create an
API key in the dashboard, and add the provider to `opencode.json` as in
`examples/opencode/opencode.json` (with your key and `http://127.0.0.1:8000/v1`). Remember
that opencode runs the commands the model chooses, on your machine: use a sandbox.

Without opencode, `minilab.eval.code` plays one request on a directory, the tools in the
Docker sandbox (the directory itself is copied, never modified):

```bash
uv run python -m minilab.eval.code models/mini-code-1 examples/opencode/project "Run the tests and fix any bug"
```

## Limitations

- Its world is small Python projects of a few files and little functions. Anything else
  (real code, other languages, long files, vague requests) is out of reach.
- One request at a time works best. Long sessions go through the context management, and by
  the fourth request the model can fall back on its shortcuts (fixing the line that looks like
  a bug).
- It runs whatever it decides to run: keep it in a sandbox.
- It speaks opencode 2's tool formats; opencode 1.x differs.
