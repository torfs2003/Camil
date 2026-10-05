"""Extract article-only chunks from the Belgian Codex PDF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


ARTICLE_MARKER = re.compile(
    r"(?m)^\s*Art\.\s*([IVXLCDM]+\.\d+-\d+[a-z]?)\b",
    re.IGNORECASE,
)
APPENDIX_MARKER = re.compile(r"(?m)^\s*BIJLAGE\b")
CHAPTER_HEADING = re.compile(r"(?im)^\s*Hoofdstuk\s+[IVXLCDM]+\s*\.-.*$")
ARTICLE_NUMBER_IN_QUESTION = re.compile(
    r"\b(?:art(?:ikel)?\.?\s*)?([IVXLCDM]+\.\d+-\d+[a-z]?)\b",
    re.IGNORECASE,
)
PAGE_NOISE = re.compile(
    r"^\s*(?:pagina\s+\d+|"
    r"(?:codex over het welzijn op het werk|"
    r"federale overheidsdienst werkgelegenheid, arbeid en sociaal overleg|"
    r"algemene directie humanisering van de arbeid)(?:\s+\d+)?)\s*$",
    re.IGNORECASE,
)
SECTION_HEADING = re.compile(
    r"(?im)^\s*(?:Boek\s+[IVXLCDM]+|Hoofdstuk\s+[IVXLCDM]+|"
    r"Titel\s+\d+|Afdeling\s+(?:\d+|[IVXLCDM]+))\s*[.\-–—].*$"
)
AMENDMENT_NOTE = re.compile(r"(?im)^\s*Gewijzigd bij\s*:")


PARAGRAPH_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<label>§[ \t]*\d+[a-z]?[ \t]*\.?|\d{1,2}\.)[ \t]*"
)
ITEM_MARKER = re.compile(r"(?m)^[ \t]*(?P<label>\d{1,2}[ \t]*[°º])[ \t]*")


@dataclass(frozen=True)
class Article:
    number: str
    text: str
    page_start: int
    page_end: int
    section_context: str = ""


@dataclass(frozen=True)
class ArticlePart:
    index: int
    label: str
    text: str
    start: int
    end: int
    search_text: str


def _clean_text(text: str) -> str:
    raw_lines = text.splitlines()
    lines: list[str] = []
    skip_page_number = False
    for line in raw_lines:
        stripped = line.strip()
        if skip_page_number and stripped.isdigit():
            skip_page_number = False
            continue
        skip_page_number = False
        if stripped.casefold() == "pagina":
            skip_page_number = True
            continue
        if not stripped or PAGE_NOISE.match(stripped):
            continue
        lines.append(stripped)
    cleaned = "\n".join(lines)
    # Section headings and amendment notes after an article belong to the
    # next section's document furniture, not to the preceding article chunk.
    boundaries = [
        match
        for pattern in (SECTION_HEADING, AMENDMENT_NOTE)
        if (match := pattern.search(cleaned)) is not None
    ]
    if boundaries:
        cleaned = cleaned[:min(boundaries, key=lambda match: match.start()).start()]
    # Remove common PDF footer fragments split into separate extracted lines.
    cleaned = re.sub(r"(?im)^\s*versie van\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d{1,2}/\d{1,2}/\d{4}\s*$", "", cleaned)
    # This PDF maps several printed symbols to the replacement character:
    # digit + � is an ordinal/degree sign; � + number is a paragraph sign.
    cleaned = re.sub(r"(?<=\d)�", "°", cleaned)
    cleaned = re.sub(r"(?<!\S)�(?=\s*\d)", "§", cleaned)
    # Quotation marks around standard names are also unmapped in the PDF text layer.
    cleaned = re.sub(r"�(.{1,300}?)�", r"“\1”", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()

def read_articles(pdf_path: str | Path) -> list[Article]:
    """Extract article-only chunks and carry section headings as search context."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF niet gevonden: {path}")

    reader = PdfReader(str(path))
    articles: list[Article] = []
    current_number: str | None = None
    current_parts: list[str] = []
    current_start = 0
    current_end = 0
    current_section_context = ""
    active_section_context = ""
    active_section_parts: dict[str, str] = {}
    seen: set[str] = set()
    skipping_appendix = False

    def finish_current() -> None:
        nonlocal current_number, current_parts, current_start, current_end
        nonlocal current_section_context
        if current_number is not None:
            body = _clean_text("\n".join(current_parts))
            if body:
                articles.append(
                    Article(
                        current_number,
                        body,
                        current_start,
                        current_end,
                        current_section_context,
                    )
                )
        current_number = None
        current_parts = []
        current_start = 0
        current_end = 0
        current_section_context = ""

    def carry_headings(headings: list[re.Match[str]]) -> None:
        nonlocal active_section_context
        levels = {"boek": 0, "titel": 1, "hoofdstuk": 2, "afdeling": 3}
        for heading in headings:
            text = re.sub(r"\s+", " ", heading.group(0)).strip()
            match = re.match(r"(Boek|Titel|Hoofdstuk|Afdeling)\b", text, re.IGNORECASE)
            if not match:
                continue
            kind = match.group(1).casefold()
            level = levels[kind]
            for existing in list(active_section_parts):
                if levels[existing] >= level:
                    del active_section_parts[existing]
            active_section_parts[kind] = text
            active_section_context = " | ".join(
                active_section_parts[key]
                for key in ("boek", "titel", "hoofdstuk", "afdeling")
                if key in active_section_parts
            )

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        markers = list(ARTICLE_MARKER.finditer(page_text))

        if skipping_appendix:
            if not markers:
                continue
            # Resume at the next article, leaving appendix text and page headings out.
            page_text = page_text[markers[0].start():]
            markers = list(ARTICLE_MARKER.finditer(page_text))
            skipping_appendix = False

        appendix = APPENDIX_MARKER.search(page_text)
        if appendix:
            page_text = page_text[:appendix.start()]
            markers = [marker for marker in markers if marker.start() < appendix.start()]

        headings = list(SECTION_HEADING.finditer(page_text))
        if not markers:
            if current_number is not None and _clean_text(page_text):
                current_parts.append(page_text)
                current_end = page_number
            carry_headings(headings)
            if appendix:
                finish_current()
                skipping_appendix = True
                active_section_context = ""
                active_section_parts.clear()
            continue

        # Text before the first article continues the preceding article,
        # except page furniture and headings removed by _clean_text.
        prefix = page_text[:markers[0].start()]
        if current_number is not None and _clean_text(prefix):
            current_parts.append(prefix)
            current_end = page_number

        heading_index = 0
        for marker_index, marker in enumerate(markers):
            # Headings encountered between articles describe the following article.
            while heading_index < len(headings) and headings[heading_index].start() < marker.start():
                carry_headings([headings[heading_index]])
                heading_index += 1

            finish_current()
            number = marker.group(1).upper()
            segment_end = (
                markers[marker_index + 1].start()
                if marker_index + 1 < len(markers)
                else len(page_text)
            )
            raw_body = page_text[marker.end():segment_end]
            raw_body = re.sub(r"^[\s.�§-]+", "", raw_body)
            if number in seen:
                continue
            seen.add(number)
            current_number = number
            current_parts = [raw_body]
            current_start = page_number
            current_end = page_number
            current_section_context = active_section_context

        # A heading after the final article on this page is for a later article.
        carry_headings(headings[heading_index:])
        if appendix:
            finish_current()
            skipping_appendix = True
            active_section_context = ""
            active_section_parts.clear()

    finish_current()
    return articles

def find_article_number(question: str) -> str | None:
    match = ARTICLE_NUMBER_IN_QUESTION.search(question)
    return match.group(1).upper() if match else None


def split_article_into_parts(article: Article) -> list[ArticlePart]:
    """Split an article into searchable numbered passages and retain full-text offsets."""
    text = article.text
    parts: list[ArticlePart] = []

    def add_part(
        start: int,
        end: int,
        label: str,
        context: tuple[str, ...] = (),
    ) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        passage = text[start:end]
        if not passage:
            return

        search_sections = [f"Art. {article.number}."]
        if article.section_context:
            search_sections.append(article.section_context)
        search_sections.extend(value[:700] for value in context if value and value != passage)
        search_sections.append(passage)
        parts.append(
            ArticlePart(
                index=len(parts),
                label=label.strip() or "Onderdeel",
                text=passage,
                start=start,
                end=end,
                search_text="\n".join(search_sections),
            )
        )

    paragraph_markers = list(PARAGRAPH_MARKER.finditer(text))
    if paragraph_markers:
        preamble_end = paragraph_markers[0].start()
        preamble = text[:preamble_end].strip()
        if preamble:
            add_part(0, preamble_end, "Inleiding")

        for paragraph_index, paragraph in enumerate(paragraph_markers):
            paragraph_start = paragraph.start("label")
            paragraph_end = (
                paragraph_markers[paragraph_index + 1].start()
                if paragraph_index + 1 < len(paragraph_markers)
                else len(text)
            )
            items = list(ITEM_MARKER.finditer(text, paragraph.end(), paragraph_end))
            if not items:
                add_part(
                    paragraph_start,
                    paragraph_end,
                    paragraph.group("label"),
                    (preamble,),
                )
                continue

            paragraph_intro = text[paragraph_start:items[0].start()].strip()
            for item_index, item in enumerate(items):
                item_end = (
                    items[item_index + 1].start()
                    if item_index + 1 < len(items)
                    else paragraph_end
                )
                add_part(
                    item.start("label"),
                    item_end,
                    item.group("label"),
                    (preamble, paragraph_intro),
                )
    else:
        items = list(ITEM_MARKER.finditer(text))
        if not items:
            add_part(0, len(text), "Artikel")
        else:
            preamble_end = items[0].start()
            preamble = text[:preamble_end].strip()
            if preamble:
                add_part(0, preamble_end, "Inleiding")
            for item_index, item in enumerate(items):
                item_end = items[item_index + 1].start() if item_index + 1 < len(items) else len(text)
                add_part(
                    item.start("label"),
                    item_end,
                    item.group("label"),
                    (preamble,),
                )

    return parts


def article_documents(articles: list[Article]) -> list[Document]:
    """Create one vector-store document per structured article passage."""
    documents: list[Document] = []
    for article in articles:
        for part in split_article_into_parts(article):
            documents.append(
                Document(
                    page_content=part.search_text,
                    metadata={
                        "article_number": article.number,
                        "part_index": part.index,
                        "part_label": part.label,
                        "page_start": article.page_start,
                        "page_end": article.page_end,
                        "section_context": article.section_context,
                    },
                )
            )
    return documents
