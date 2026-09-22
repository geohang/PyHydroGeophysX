"""Markdown building blocks for a report that reads like a deliverable.

The reports were assembled as runs of bullet points and raw ``json.dumps``
blocks. That is legible to whoever wrote the workflow and to nobody else: a
reader looking for the final chi-squared had to find it inside a fenced block
of three hundred lines, and a five-survey inventory arrived as a JSON array
carrying absolute paths and full SHA-256 digests.

Nothing here changes what a report says. It changes where a reader has to look
for it: a document-control block at the top, numbered sections, tabulated
numbers, and the machine-readable record moved to appendices with the complete
version left in ``processing_audit.json``.

The one formatting rule worth stating: every value that reaches a table passes
through :func:`cell`, because an unescaped pipe or a newline inside a value
silently destroys the row it sits in - and Windows paths, free-text warnings
and model-written sentences are exactly the values most likely to carry one.
"""

from typing import Any, Iterable, List, Optional, Sequence

#: Digests are quoted at this length in the document; the full value stays in
#: ``processing_audit.json``. Twelve hex characters distinguish the survey
#: files of a run by eye without wrapping a table column.
DIGEST_CHARS = 12


def cell(value: Any) -> str:
    """One table cell: never empty, never breaking the row.

    Parameters
    ----------
    value : any
        Rendered with ``str``. None and blank become an em dash, so a column
        cannot silently collapse.

    Returns
    -------
    str
        Text safe to place between two pipes.

    Raises
    ------
    None

    Examples
    --------
    >>> cell('C:/a|b')
    'C:/a\\\\|b'
    >>> cell(None)
    '\u2014'
    >>> cell('two\\nlines')
    'two lines'
    """
    if value is None:
        return "\u2014"
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return "\u2014"
    return text.replace("|", "\\|")


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]],
          align: Optional[Sequence[str]] = None) -> str:
    """A GitHub-flavoured markdown table, or ``""`` when there are no rows.

    Parameters
    ----------
    headers : sequence of str
        Column headings.
    rows : iterable of sequence
        One sequence per row; rows shorter than ``headers`` are padded.
    align : sequence of str, optional
        ``'left'``, ``'right'`` or ``'center'`` per column, left by default.
        Numeric columns read better right-aligned.

    Returns
    -------
    str
        The table ending in a newline, or an empty string when there are no
        rows - so a caller can concatenate it without leaving a heading
        standing over nothing.

    Raises
    ------
    None

    Examples
    --------
    >>> print(table(['Step', 'Chi-squared'], [[1, 2461.282], [2, 1.63]],
    ...             align=['left', 'right']))
    | Step | Chi-squared |
    | :-- | --: |
    | 1 | 2461.282 |
    | 2 | 1.63 |
    <BLANKLINE>
    >>> table(['Step'], [])
    ''
    """
    rows = [list(row) for row in rows]
    if not rows:
        return ""
    markers = {"left": ":--", "right": "--:", "center": ":-:"}
    rule = [markers.get((align or [])[i] if align and i < len(align) else "left",
                        ":--")
            for i in range(len(headers))]
    lines = ["| " + " | ".join(cell(h) for h in headers) + " |",
             "| " + " | ".join(rule) + " |"]
    for row in rows:
        padded = list(row) + [None] * (len(headers) - len(row))
        lines.append("| " + " | ".join(cell(v) for v in padded[:len(headers)]) + " |")
    return "\n".join(lines) + "\n"


def facts(pairs: Iterable[Sequence[Any]], label: str = "Item",
          value: str = "Value") -> str:
    """A two-column table for a block of labelled values.

    Parameters
    ----------
    pairs : iterable of (str, any)
        Label and value. A pair whose value is None is kept and shows an em
        dash: an absent value is information, and dropping the row hides that
        the field was meant to be there at all.
    label, value : str
        Column headings.

    Returns
    -------
    str
        The table, or ``""`` when ``pairs`` is empty.

    Raises
    ------
    None

    Examples
    --------
    >>> print(facts([('Site', 'Medicine Bow'), ('Coordinates', None)]))
    | Item | Value |
    | :-- | :-- |
    | Site | Medicine Bow |
    | Coordinates | \u2014 |
    <BLANKLINE>
    """
    return table([label, value], [[a, b] for a, b in pairs])


def digest(sha256: Optional[str]) -> str:
    """A digest shortened for reading, and marked as abbreviated.

    Parameters
    ----------
    sha256 : str or None
        The full hexadecimal digest.

    Returns
    -------
    str
        The first :data:`DIGEST_CHARS` characters followed by an ellipsis, or
        an em dash. The ellipsis earns its place: an unmarked truncation looks
        like a digest someone can compare, and it is not one.

    Raises
    ------
    None

    Examples
    --------
    >>> digest('38ce1b36f3235cec1c18f580f03a31d0')
    '38ce1b36f323\u2026'
    >>> digest(None)
    '\u2014'
    """
    if not sha256:
        return "\u2014"
    text = str(sha256)
    return text[:DIGEST_CHARS] + "\u2026" if len(text) > DIGEST_CHARS else text


def file_size(nbytes: Optional[int]) -> str:
    """A byte count as a person would write it.

    Parameters
    ----------
    nbytes : int or None
        Size in bytes.

    Returns
    -------
    str
        For example ``'176.2 kB'``, or an em dash when unknown.

    Raises
    ------
    None

    Examples
    --------
    >>> file_size(180456)
    '176.2 kB'
    >>> file_size(343)
    '343 B'
    >>> file_size(None)
    '\u2014'
    """
    if nbytes is None:
        return "\u2014"
    size = float(nbytes)
    if size < 1024:
        return f"{int(size)} B"
    for unit in ("kB", "MB", "GB"):
        size /= 1024.0
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
    return f"{size:.1f} GB"


def control_block(title: str, pairs: Iterable[Sequence[Any]],
                  notice: str = "") -> str:
    """Front matter: title, document-control table, and a standing notice.

    Parameters
    ----------
    title : str
        Report title, rendered as the document's single top-level heading.
    pairs : iterable of (str, any)
        Document-control fields - site, dates, run identifier and so on.
    notice : str, optional
        A sentence placed under the table, for the standing statement of what
        the document is and is not.

    Returns
    -------
    str
        Markdown ending in a newline.

    Raises
    ------
    None

    Examples
    --------
    >>> block = control_block('Site Report', [('Run', 'abc')], notice='Draft.')
    >>> block.splitlines()[0]
    '# Site Report'
    >>> '| Field | Detail |' in block
    True
    >>> 'Draft.' in block
    True
    """
    parts = [f"# {title}", "", facts(pairs, label="Field", value="Detail")]
    if notice:
        parts += ["", notice]
    return "\n".join(parts) + "\n"


def bullets(items: Iterable[Any]) -> str:
    """A bullet list, or ``""`` when there is nothing to list.

    Parameters
    ----------
    items : iterable
        Rendered with ``str``; blank entries are dropped.

    Returns
    -------
    str
        The list ending in a newline, or an empty string.

    Raises
    ------
    None

    Examples
    --------
    >>> print(bullets(['first', '', 'second']))
    - first
    - second
    <BLANKLINE>
    >>> bullets([])
    ''
    """
    lines: List[str] = [f"- {str(item).strip()}" for item in items
                        if str(item).strip()]
    return "\n".join(lines) + "\n" if lines else ""


def numbered(items: Iterable[Any]) -> str:
    """A numbered list, or ``""`` when there is nothing to list.

    Parameters
    ----------
    items : iterable
        Rendered with ``str``; blank entries are dropped.

    Returns
    -------
    str
        The list ending in a newline, or an empty string.

    Raises
    ------
    None

    Examples
    --------
    >>> print(numbered(['first', 'second']))
    1. first
    2. second
    <BLANKLINE>
    >>> numbered([])
    ''
    """
    kept = [str(item).strip() for item in items if str(item).strip()]
    if not kept:
        return ""
    return "\n".join(f"{i}. {text}" for i, text in enumerate(kept, 1)) + "\n"


def renumber(markdown: str) -> str:
    """Number the ``##`` and ``###`` headings of a document in order.

    Section numbers are applied here rather than written into each heading
    because the body is assembled from parts that appear conditionally - a
    climate section only when climate data were retrieved, a water-content
    section only when the conversion ran. Hand-numbered headings drift the
    first time one of those is skipped, and a report whose sections run 4, 6, 7
    reads as one with a section missing.

    Headings from the first one beginning with "Appendix" onward are left as
    written: appendices carry their own letters. Text inside fenced code blocks
    is never touched.

    Parameters
    ----------
    markdown : str
        The assembled document.

    Returns
    -------
    str
        The same document with ``## Heading`` renumbered to ``## 1. Heading``
        and ``### Heading`` to ``### 1.1 Heading``. A heading that already
        carries a number is renumbered rather than numbered twice.

    Raises
    ------
    None

    Examples
    --------
    >>> print(renumber('## Scope\\n### Data\\n### Method\\n## Results\\n'))
    ## 1. Scope
    ### 1.1 Data
    ### 1.2 Method
    ## 2. Results
    <BLANKLINE>
    >>> print(renumber('## Results\\n## Appendix A - Inputs\\n### Files\\n'))
    ## 1. Results
    ## Appendix A - Inputs
    ### Files
    <BLANKLINE>
    """
    out: List[str] = []
    section = 0
    subsection = 0
    in_fence = False
    in_appendices = False
    for line in markdown.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or in_appendices:
            out.append(line)
            continue
        if line.startswith("## ") and not line.startswith("### "):
            title = _strip_number(line[3:])
            if title.lower().startswith("appendix"):
                in_appendices = True
                out.append("## " + title)
                continue
            section += 1
            subsection = 0
            out.append(f"## {section}. {title}")
        elif line.startswith("### ") and not line.startswith("#### "):
            title = _strip_number(line[4:])
            if section == 0:
                out.append("### " + title)
            else:
                subsection += 1
                out.append(f"### {section}.{subsection} {title}")
        else:
            out.append(line)
    return "\n".join(out)


def _strip_number(title: str) -> str:
    """A heading without a number a previous pass may have given it."""
    text = title.strip()
    head = text.split(" ", 1)
    if len(head) == 2 and head[0].rstrip(".").replace(".", "").isdigit():
        return head[1].strip()
    return text
