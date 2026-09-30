"""A sandbox with opencode's coding tools, for generating mini-code's training data and for
evaluating it.

Each tool takes the arguments of opencode's own tool (v2: `read`, `write`, `edit`, `glob`,
`grep`, `shell`) and returns *exactly* the text opencode sends back to the model, so a model
trained on these transcripts sees the same thing when it runs inside opencode. The formats
were captured from opencode 2.0.20 with a fake OpenAI server (see docs/opencode.md):

    read   Read file calc.py, lines 1-2\\n1: def add(a, b):\\n2:     return a + b
    write  Created file successfully: calc.py            (Wrote file successfully: ... if it existed)
    edit   Edited calc.py (1 replacement)
    glob   /abs/path/b.py\\n/abs/path/a.py                  (newest first)
    grep   Found 1 matches\\n/abs/path/calc.py:\\n  Line 1: def add(a, b):\\n
    shell  stdout and stderr, then "\\nExited with code 1" when it fails; "(no output)" if silent

A failing tool returns opencode's error JSON: {"error": {"type": "tool.execution", "message": ...},
"content": []}, which the chat template shortens to "Error: <message>".

Sandbox runs the tools on this machine: only for commands we wrote ourselves (the oracles that
generate the training data). What a *model* asks for runs in a DockerSandbox: the same tools,
inside a container with no network, a read-only system, no capabilities and a memory, CPU and
process limit, so a model's `rm -rf ~` or `curl` can't reach the host. This file only uses the
standard library: the container runs it as is (`python3 sandbox.py serve`).
"""

from __future__ import annotations

import fnmatch
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

# opencode v2's tools, in the order it lists them in every request. mini-code only ever calls
# the first six; the others are there because opencode always offers them.
OPENCODE_TOOLS = ["edit", "glob", "grep", "question", "read", "shell", "skill", "subagent", "webfetch",
                  "websearch", "write", "execute"]
CODE_TOOLS = ["read", "write", "edit", "glob", "grep", "shell"]
# The argument types of opencode's JSON schemas that aren't strings: the chat template writes
# every argument as text, and they are typed back with these (in the API: the client's schemas).
TOOL_SCHEMAS = {
    "read": {"properties": {"offset": {"type": "integer"}, "limit": {"type": "integer"}}},
    "edit": {"properties": {"replaceAll": {"type": "boolean"}}},
    "glob": {"properties": {"hidden": {"type": "boolean"}, "limit": {"type": "integer"}}},
    "grep": {"properties": {"literal": {"type": "boolean"}, "caseSensitive": {"type": "boolean"},
                            "limit": {"type": "integer"}}},
    "shell": {"properties": {"timeout": {"type": "integer"}, "background": {"type": "boolean"}}},
}

SHELL_TIMEOUT_S = 10


class ToolError(Exception):
    pass


def error_json(message: str) -> str:
    """What opencode sends the model when a tool fails."""
    return json.dumps({"error": {"type": "tool.execution", "message": message}, "content": []}, separators=(",", ":"))


class Sandbox:
    """A project directory the tools act on. Paths in tool arguments are relative to it."""

    def __init__(self, files: dict[str, str] | None = None, root: str | Path | None = None):
        self._tmp = None
        if root is None:
            self._tmp = tempfile.mkdtemp(prefix="minicode-")
            root = self._tmp
        self.root = Path(root).resolve()
        for path, content in (files or {}).items():
            self.write_file(path, content)

    def close(self) -> None:
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- files ----------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """A path inside the project (symlinks followed); anything outside it is refused."""
        p = Path(path)
        p = (p if p.is_absolute() else self.root / p).resolve()
        if not p.is_relative_to(self.root):
            raise ToolError(f"Access denied: {path} is outside the project directory")
        return p

    def write_file(self, path: str, content: str) -> None:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def read_file(self, path: str) -> str | None:
        p = self._resolve(path)
        return p.read_text() if p.is_file() else None

    def files(self) -> dict[str, str]:
        """Every file of the project, relative path -> content."""
        return {str(p.relative_to(self.root)): p.read_text() for p in sorted(self.root.rglob("*"))
                if p.is_file() and not _hidden(p.relative_to(self.root))}

    def _newest_first(self, paths: list[Path]) -> list[Path]:
        return sorted(paths, key=lambda p: (-p.stat().st_mtime_ns, str(p)))

    # ---- tools ------------------------------------------------------------------

    def call(self, name: str, arguments: dict) -> str:
        """Run one tool call, returning what opencode would send back (errors included)."""
        tools = {"read": self.read, "write": self.write, "edit": self.edit, "glob": self.glob,
                 "grep": self.grep, "shell": self.shell}
        if name not in tools:
            return error_json(f'No tool named "{name}" is currently available. '
                              "Please use a tool from the available tool list.")
        try:
            return tools[name](**arguments)
        except ToolError as e:
            return error_json(str(e))
        except TypeError:
            return error_json(f'Invalid arguments for tool "{name}":\n'
                              f"Arguments provided:\n{json.dumps(arguments, indent=2)}\n\n"
                              "Update the arguments and call the tool again.")

    def read(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        p = self._resolve(path)
        if p.is_dir():
            entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir() if not e.name.startswith("."))
            return f"Read directory {path}, entries 1-{len(entries)}\n" + "\n".join(entries)
        if not p.is_file():
            raise ToolError(f"File not found: {path}")
        lines = p.read_text().splitlines()
        if not lines:
            return f"Read file {path}, 0 lines"
        start = max(1, int(offset or 1))
        end = min(len(lines), start + int(limit or 2000) - 1)
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        out = f"Read file {path}, lines {start}-{end}\n{body}"
        if end < len(lines):
            out += f"\n[Output truncated. Continue reading with offset: {end + 1}]"
        return out

    def write(self, path: str, content: str) -> str:
        existed = self._resolve(path).is_file()
        self.write_file(path, content)
        return f"{'Wrote' if existed else 'Created'} file successfully: {path}"

    def edit(self, path: str, oldString: str, newString: str, replaceAll: bool = False) -> str:  # noqa: N803 (opencode's names)
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"File not found: {path}")
        if oldString == newString:
            raise ToolError("No changes to apply: oldString and newString are identical.")
        text = p.read_text()
        n = text.count(oldString) if oldString else 0
        if n == 0:
            raise ToolError(f"Could not find oldString in {path}. It must match exactly, including whitespace "
                            "and indentation.")
        if n > 1 and not replaceAll:
            raise ToolError(f"Found {n} matches for oldString, but expected exactly one. Add more surrounding "
                            "context to make oldString unique, or set replaceAll to true to replace every occurrence.")
        p.write_text(text.replace(oldString, newString))
        return f"Edited {path} ({n} replacement{'s' if n > 1 else ''})"

    def glob(self, pattern: str, path: str | None = None, **_) -> str:
        base = self._resolve(path or ".")
        hits = [p for p in base.rglob("*") if p.is_file() and not _hidden(p.relative_to(base))
                and _glob_match(str(p.relative_to(base)), pattern)]
        if not hits:
            return "No files found"
        return "\n".join(str(p) for p in self._newest_first(hits)[:100])

    def grep(self, pattern: str, path: str | None = None, include: str | None = None, **_) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"Invalid regular expression: {e}") from None
        base = self._resolve(path or ".")
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()
                                                and not _hidden(p.relative_to(base))]
        if include:
            files = [p for p in files if _glob_match(p.name, include)]
        groups, total = [], 0
        for p in self._newest_first(files):
            try:
                lines = p.read_text().splitlines(keepends=True)
            except UnicodeDecodeError:
                continue
            hits = [(i, line) for i, line in enumerate(lines, 1) if regex.search(line)]
            if hits:
                total += len(hits)
                groups.append([f"{p}:", *(f"  Line {i}: {line}" for i, line in hits)])
        if not groups:
            return "No matches found"
        out = [f"Found {total} matches"]
        for i, group in enumerate(groups):
            if i:
                out.append("")
            out += group
        return "\n".join(out)

    def shell(self, command: str, workdir: str | None = None, **_) -> str:
        try:
            proc = subprocess.run(command, shell=True, cwd=self._resolve(workdir or "."), capture_output=False,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  timeout=SHELL_TIMEOUT_S, env=_shell_env())
        except subprocess.TimeoutExpired:
            return f"Command timed out after {SHELL_TIMEOUT_S * 1000} ms"
        out = proc.stdout
        if proc.returncode == 0:
            return out or "(no output)"
        return (out if not out or out.endswith("\n") else out + "\n") + f"\nExited with code {proc.returncode}"


def _hidden(rel: Path) -> bool:
    return any(part.startswith(".") or part == "__pycache__" for part in rel.parts)


def _glob_match(rel: str, pattern: str) -> bool:
    """Glob with ** spanning directories (fnmatch's * already crosses "/"; "**/" may match nothing)."""
    if pattern.startswith("**/"):
        return fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pattern[3:])
    return fnmatch.fnmatch(rel, pattern) and ("/" in pattern or "/" not in rel)


def _shell_env() -> dict:
    # No .pyc files next to the project, and no user site-packages leaking into the sandbox.
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}


# ---------------------------------------------------------------------------
# Running a model's tool calls in a container
# ---------------------------------------------------------------------------

SANDBOX_IMAGE = "python:3.13-slim"


def serve() -> None:
    """The container side: JSON requests on stdin, one per line, answers on stdout."""
    boxes: dict[str, Sandbox] = {}
    counter = itertools.count()
    for line in sys.stdin:
        req = json.loads(line)
        try:
            op = req["op"]
            if op == "new":
                root = Path("/work") / f"p{next(counter)}"
                root.mkdir(parents=True)
                box = Sandbox(req["files"], root=root)
                boxes[str(box.root)] = box
                out = {"root": str(box.root)}
            elif op == "call":
                out = {"result": boxes[req["root"]].call(req["name"], req["args"])}
            elif op == "files":
                out = {"files": boxes[req["root"]].files()}
            elif op == "close":
                shutil.rmtree(req["root"], ignore_errors=True)
                boxes.pop(req["root"], None)
                out = {}
            else:
                out = {"error": f"unknown op {op}"}
        except Exception as e:  # never let one bad call kill the server
            out = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


class Container:
    """One locked-down container running serve(); many DockerSandboxes share it."""

    def __init__(self, image: str = SANDBOX_IMAGE, memory: str = "512m", cpus: str = "2"):
        self.id = subprocess.run(
            ["docker", "run", "-d", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
             "--security-opt", "no-new-privileges", "--user", "65534:65534", "--memory", memory,
             "--cpus", cpus, "--pids-limit", "128", "--tmpfs", "/work:rw,size=256m,uid=65534,gid=65534",
             "--tmpfs", "/tmp:rw,size=64m", "-v", f"{Path(__file__).resolve()}:/opt/sandbox.py:ro",
             image, "sleep", "infinity"],
            check=True, capture_output=True, text=True).stdout.strip()
        self.proc = subprocess.Popen(["docker", "exec", "-i", self.id, "python3", "/opt/sandbox.py", "serve"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.lock = threading.Lock()

    def request(self, **req) -> dict:
        with self.lock:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("the sandbox container stopped")
        out = json.loads(line)
        if "error" in out:
            raise RuntimeError(f"sandbox: {out['error']}")
        return out

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        subprocess.run(["docker", "rm", "-f", self.id], capture_output=True)

    def __enter__(self) -> "Container":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DockerSandbox:
    """A project inside a Container, with the same interface as Sandbox."""

    def __init__(self, container: Container, files: dict[str, str] | None = None):
        self.container = container
        self.root = Path(container.request(op="new", files=files or {})["root"])

    def call(self, name: str, arguments: dict) -> str:
        return self.container.request(op="call", root=str(self.root), name=name, args=arguments)["result"]

    def shell(self, command: str) -> str:
        return self.call("shell", {"command": command})

    def files(self) -> dict[str, str]:
        return self.container.request(op="files", root=str(self.root))["files"]

    def read_file(self, path: str) -> str | None:
        return self.files().get(path)

    def close(self) -> None:
        self.container.request(op="close", root=str(self.root))

    def __enter__(self) -> "DockerSandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        serve()
