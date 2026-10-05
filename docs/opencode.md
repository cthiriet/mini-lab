# prelude in opencode

In [opencode](https://opencode.ai), the terminal coding agent, prelude is a coding agent: it
reads requests like "Run the tests and fix any bug", calls opencode's tools (`read`, `write`,
`edit`, `glob`, `grep`, `shell`) until the job is done, and says what it did. It is a
demonstration, not a coder: its whole world is small Python projects of a few files and
little functions (`add`, `greet`, `reverse`...). The point is that tool use, the format a real
agent harness speaks and a loop of calls and results can be learned by a model this small, and
served through the same OpenAI-compatible API as any other model.

```bash
bash speedrun.sh                              # trains and releases models/prelude-2
bash examples/opencode/demo.sh                # opencode's TUI on a demo project, in containers
bash examples/opencode/demo.sh run --auto "Run the tests and fix any bug"
```

How the model learns it (the toy code world, transcripts played for real, the coding eval in
a Docker sandbox) is in [training.md](training.md); this page is about what it sees.

## One model, two jobs

Nothing tells the model which job it is doing but what the client sends, as with any assistant
model. opencode sends its system prompt (whose first sentence the model sees: "You are an AI
agent running in OpenCode, a coding agent harness.") and the line `tools: edit, glob, grep,
read, shell, ...`. The chat app sends no system prompt (or one of the trained instructions) and
`tools: calculator`, or no tools. So "Who are you?" gets "I'm prelude, ... I can list, read,
run, create, edit and fix the Python files in this project." in opencode, and "I'm prelude, a
very small language model trained from scratch on a laptop by mini-lab." in the chat; "Write a
Python function that sorts a list." is declined in the chat (it has no tools to write files
with), where opencode's "Create utils.py with a function that..." is done.

## In opencode

`bash examples/opencode/demo.sh run --auto "Run the tests and fix any bug"`, on the demo project
whose `add` returns `a - b`, next to a `multiply` that returns `a * b`:

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

prelude-2 does 98% of the coding eval's tasks end to end (new projects, greedy,
the tool calls in the Docker sandbox), and every one of its tool calls is one opencode
accepts; small talk and session titles are 100%:

| task (50 each) | success | | task (50 each) | success |
|---|---:|---|---|---:|
| list_files | 100% | | create_script | 100% |
| find_def | 100% | | rename | 100% |
| show_file | 100% | | change_const | 100% |
| explain | 100% | | add_func | 100% |
| run | 100% | | fix_test | 98% |
| run_tests | 100% | | fix_crash | 94% |
| create_func | 100% | | fix_distractor | 86% |

`fix_distractor` is the demo's trap: a failing test with, in the same module, a function whose
correct code looks like the bug (multiply's `a * b` next to a broken add). The failures left
are mostly that trap and copies, like a misspelled name to copy from a traceback.

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
- A separate request for the session's title: no tools, a "You are a title generator" system
  prompt, and the user's message.
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
formats): the model is trained on opencode 2's.

## Fitting an agent into 1,024 tokens

Small open models do the same: the chat template decides what the model sees
(`minilab/tokenizer/chat.py`):

- **Tools by name only.** The model learned opencode's tools in its weights; the line
  `tools: edit, glob, ...` tells it which are there, and the JSON schemas are dropped (as for
  the calculator).
- **The system prompt's first sentence.** A 5.8M model can't read 5,000 tokens of rules; it
  was trained on the behavior instead. The title generator's prompt becomes "You are a title
  generator."
- **Paths relative to the working directory**, taken from the system prompt's `Working
  directory:` line: `/Users/me/project/calc.py` becomes `calc.py`. Tool errors in opencode's
  JSON become `Error: <message>`.
- **Raw arguments.** A tool call is `<|tool_call_start|>edit<|arg|>path=calc.py<|arg|>oldString=    return a - b<|arg|>newString=    return a + b<|tool_call_end|>`:
  code goes in as it is, with its quotes and newlines, where JSON would need `\"` and `\n`
  (Qwen3-Coder moved from JSON to XML-style tool calls for the same reason). The API turns it
  into ordinary `tool_calls`, and types the values with the request's JSON schemas (`"true"`
  becomes `true` where the schema says boolean).
- **Server-side context management.** opencode compacts a session when it nears the model's
  context, by asking the model for a summary, which a model this small can't write. So the
  inference server keeps every prompt within the context minus the room for the answer, by
  dropping the oldest turns, then cutting the current request's tool results (see
  [inference.md](inference.md)), and `opencode.json` declares a large context so opencode
  never compacts.
- **Greedy by default.** opencode sends no temperature, and OpenAI's default is 1, too hot for
  a tiny model copying code: the release sets `"default_temperature": 0`, like a Hugging Face
  `generation_config.json`. The chat app sends its own (0.6).

## Trying it

`examples/opencode/` runs everything in containers: the inference server and API, and
opencode 2.0.20 with the demo project. Their network is internal: opencode, and whatever the
model runs through its `shell` tool, can reach the mini-lab API and nothing else. `demo.sh`
starts it with a fresh internal token; the API container creates a local account with
credits and hands its key to opencode.

To use your own opencode instead, serve the model (`./scripts/serve.sh`), create an API key
in the dashboard, and add the provider to `opencode.json` as in
`examples/opencode/opencode.json` (with your key and `http://127.0.0.1:8000/v1`). Remember that
opencode runs the commands the model chooses, on your machine: use a sandbox.

Without opencode, `minilab.eval.code` plays one request on a directory, the tools in the Docker
sandbox (the directory is copied, never modified):

```bash
uv run python -m minilab.eval.code models/prelude-2 examples/opencode/project "Run the tests and fix any bug"
```

## Limitations

- Its world is small Python projects of a few files and little functions. Anything else
  (real code, other languages, long files, vague requests) is out of reach.
- One request at a time works best. Long sessions go through the context management, and by
  the fourth request the model can fall back on its shortcuts (fixing the line that looks like
  a bug).
- It runs whatever it decides to run: keep it in a sandbox.
- It speaks opencode 2's tool formats; opencode 1.x differs.
