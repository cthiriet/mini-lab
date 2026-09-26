"""A tiny, safe Markdown renderer for model cards.

Supports what minilab.eval.model_card writes: headings, paragraphs, bullet lists,
tables (with alignment), fenced code blocks, blockquotes, `code`, **bold** and
*italic*. All text is HTML-escaped *before* any formatting is applied, so a model
card can never inject markup into the page. Anything else renders as plain text.
"""

from __future__ import annotations

import html
import re

_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"), r"<em>\1</em>"),
]


def _inline(text: str) -> str:
    # Split out code spans first, so ** and * inside them stay literal.
    out = []
    for i, part in enumerate(re.split(r"(`[^`]+`)", html.escape(text, quote=False))):
        if i % 2:
            out.append(f"<code>{part[1:-1]}</code>")
        else:
            part = part.replace("\\*", "&#42;")  # an escaped \* is a literal asterisk
            for pattern, repl in _INLINE:
                part = pattern.sub(repl, part)
            out.append(part)
    return "".join(out)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table(lines: list[str]) -> str:
    head, align_row, rows = _cells(lines[0]), _cells(lines[1]), [_cells(line) for line in lines[2:]]
    aligns = ["right" if a.endswith(":") and not a.startswith(":") else "center" if a.startswith(":") and a.endswith(":")
              else "left" for a in align_row]

    def row(cells: list[str], tag: str) -> str:
        return "<tr>" + "".join(f'<{tag} style="text-align:{aligns[i] if i < len(aligns) else "left"}">{_inline(c)}</{tag}>'
                                for i, c in enumerate(cells)) + "</tr>"

    thead = row(head, "th") if any(head) else ""  # "| | |" headers are just a key/value layout
    return f"<table><thead>{thead}</thead><tbody>{''.join(row(r, 'td') for r in rows)}</tbody></table>"


def render_markdown(text: str, skip_title: bool = False) -> str:
    """Markdown -> HTML. skip_title drops a leading `# Title` (the page already shows it)."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    if skip_title:
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines) and lines[i].startswith("# "):
            i += 1
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(lines[i + 1:j]), quote=False)}</code></pre>")
            i = j + 1
        elif m := re.match(r"(#{1,6}) (.*)", line):
            level = min(len(m.group(1)) + 1, 6)  # the page owns <h1>
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
        elif line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?\s*$", lines[i + 1]):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(_table(lines[i:j]))
            i = j
        elif re.match(r"[-*] ", line):
            items = []
            while i < len(lines) and re.match(r"[-*] ", lines[i]):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
        elif line.startswith(">"):
            quoted = []
            while i < len(lines) and lines[i].startswith(">"):
                quoted.append(lines[i][1:].removeprefix(" "))
                i += 1
            paragraphs = "\n".join(quoted).split("\n\n")
            body = "".join(f"<p>{'<br>'.join(_inline(l) for l in p.splitlines())}</p>" for p in paragraphs if p.strip())
            out.append(f"<blockquote>{body}</blockquote>")
        else:
            para = [line]  # always consume at least this line (e.g. a "|" line that isn't a table)
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"(#{1,6} |```|\||[-*] |>)", lines[i]):
                para.append(lines[i])
                i += 1
            out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)
