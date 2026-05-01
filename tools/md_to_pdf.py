#!/usr/bin/env python3
"""Convert a Chinese markdown file to a nicely-formatted PDF using WeasyPrint.

Used for generating printable / shareable versions of memory files.
"""
import argparse
import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

CSS_STYLE = """
@page {
    size: A4;
    margin: 18mm 16mm 20mm 16mm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: "Noto Sans CJK SC", sans-serif;
        font-size: 9pt;
        color: #888;
    }
}
body {
    font-family: "Noto Sans CJK SC", "Noto Sans CJK", sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #222;
}
h1 {
    font-size: 20pt;
    color: #1a4480;
    border-bottom: 2px solid #1a4480;
    padding-bottom: 6px;
    margin-top: 28px;
    margin-bottom: 14px;
    page-break-before: auto;
}
h1:first-of-type {
    page-break-before: avoid;
    margin-top: 8px;
}
h2 {
    font-size: 15pt;
    color: #2c5aa0;
    margin-top: 22px;
    margin-bottom: 10px;
    border-left: 4px solid #2c5aa0;
    padding-left: 10px;
}
h3 {
    font-size: 12.5pt;
    color: #444;
    margin-top: 16px;
    margin-bottom: 8px;
}
p {
    margin: 8px 0;
    text-align: justify;
}
strong {
    color: #b54708;
    font-weight: 700;
}
em {
    color: #555;
}
code {
    background-color: #f4f5f7;
    padding: 1px 4px;
    border-radius: 3px;
    font-family: "DejaVu Sans Mono", "Noto Sans CJK SC", monospace;
    font-size: 9.5pt;
    color: #c7254e;
}
pre {
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    padding: 10px 12px;
    font-family: "DejaVu Sans Mono", "Noto Sans CJK SC", monospace;
    font-size: 9pt;
    line-height: 1.4;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
}
pre code {
    background: none;
    padding: 0;
    color: #24292e;
}
blockquote {
    border-left: 3px solid #c6cbd1;
    padding: 4px 12px;
    margin: 10px 0;
    color: #586069;
    background: #f6f8fa;
    font-style: italic;
}
table {
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 9.5pt;
    width: 100%;
}
th, td {
    border: 1px solid #d0d7de;
    padding: 5px 8px;
    text-align: left;
    vertical-align: top;
}
th {
    background-color: #f0f4f8;
    font-weight: 600;
    color: #1a4480;
}
tr:nth-child(even) {
    background-color: #fafbfc;
}
ul, ol {
    margin: 6px 0;
    padding-left: 24px;
}
li {
    margin: 3px 0;
}
hr {
    border: none;
    border-top: 1px solid #d0d7de;
    margin: 22px 0;
}
.frontmatter {
    background: #fff8e1;
    border-left: 3px solid #f0a020;
    padding: 8px 12px;
    margin: 10px 0 18px 0;
    font-size: 9pt;
    color: #555;
}
.title-page {
    text-align: center;
    margin-top: 40mm;
    margin-bottom: 30mm;
}
.title-page h1 {
    border: none;
    color: #1a4480;
    font-size: 24pt;
    margin-bottom: 8mm;
}
.title-page .subtitle {
    font-size: 13pt;
    color: #555;
    margin-bottom: 4mm;
}
.title-page .meta {
    font-size: 10pt;
    color: #888;
    margin-top: 12mm;
}
"""


def strip_frontmatter(text: str):
    """Remove YAML frontmatter, return (frontmatter_html, body_md)."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            fm = text[4:end]
            body = text[end + 5:]
            # Render frontmatter as a small info block
            fm_html = '<div class="frontmatter"><strong>Frontmatter:</strong><br>'
            for line in fm.splitlines():
                if line.strip():
                    fm_html += line.replace("<", "&lt;").replace(">", "&gt;") + "<br>"
            fm_html += "</div>"
            return fm_html, body
    return "", text


def make_title_page(title: str, subtitle: str = "", date: str = ""):
    return f"""
<div class="title-page">
    <h1>{title}</h1>
    {f'<div class="subtitle">{subtitle}</div>' if subtitle else ''}
    <div class="meta">{date}</div>
</div>
<div style="page-break-after: always;"></div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_md", help="Path to .md file")
    ap.add_argument("output_pdf", help="Path to output .pdf")
    ap.add_argument("--title", default=None, help="Title for cover page")
    ap.add_argument("--subtitle", default="", help="Subtitle")
    ap.add_argument("--date", default="", help="Date string")
    ap.add_argument("--no-frontmatter-block", action="store_true",
                    help="Hide YAML frontmatter info block")
    args = ap.parse_args()

    src = Path(args.input_md).read_text(encoding="utf-8")
    fm_html, body_md = strip_frontmatter(src)

    md = markdown.Markdown(extensions=[
        "tables", "fenced_code", "extra", "sane_lists", "nl2br",
    ])
    body_html = md.convert(body_md)

    title = args.title or Path(args.input_md).stem
    cover = make_title_page(title, args.subtitle, args.date)

    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
{cover}
{'' if args.no_frontmatter_block else fm_html}
{body_html}
</body>
</html>
"""

    fc = FontConfiguration()
    HTML(string=full_html).write_pdf(
        args.output_pdf,
        stylesheets=[CSS(string=CSS_STYLE, font_config=fc)],
        font_config=fc,
    )
    print(f"PDF written to {args.output_pdf}")


if __name__ == "__main__":
    main()
