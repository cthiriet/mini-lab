"""The model-card Markdown renderer: the subset we write, and never raw HTML."""

from minilab.platform.markdown import render_markdown


def test_renders_the_model_card_subset():
    html = render_markdown(
        "# mini-1\n\nIntro with `code`, **bold** and *italic*.\n\n## Model\n\n"
        "| stage | steps |\n|---|---:|\n| pretrain | 3,500 |\n\n- one\n- two\n\n"
        "```\nraw  text\n```\n\n> **Hi!**\n>\n> Hello!\n> How are you?\n", skip_title=True)
    assert "<h1>" not in html and "mini-1" not in html  # the page already shows the title
    assert "<code>code</code>" in html and "<strong>bold</strong>" in html and "<em>italic</em>" in html
    assert "<h3>Model</h3>" in html
    assert '<th style="text-align:right">steps</th>' in html and "<td style=\"text-align:left\">pretrain</td>" in html
    assert "<ul><li>one</li><li>two</li></ul>" in html
    assert "<pre><code>raw  text</code></pre>" in html
    assert "<blockquote><p><strong>Hi!</strong></p><p>Hello!<br>How are you?</p></blockquote>" in html


def test_escapes_html_everywhere():
    html = render_markdown("<script>alert(1)</script>\n\n| <b>x</b> | y |\n|---|---|\n| `<i>` | **<u>** |\n\n```\n<img src=x>\n```")
    assert "<script>" not in html and "<b>" not in html and "<i>" not in html and "<u>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html


def test_never_loops_on_odd_lines():
    assert render_markdown("| not a table\n#nohash\n-nodash") == "<p>| not a table #nohash -nodash</p>"
