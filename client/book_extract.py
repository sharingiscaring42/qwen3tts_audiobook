import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split()).strip()


def _chapter_summary(index: int, title: str, text: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    return {
        "index": index,
        "title": title,
        "text": normalized,
        "word_count": len(normalized.split()),
        "preview": normalized[:100],
    }


def _flatten_toc(toc: List[Any]) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for item in toc:
        if isinstance(item, list):
            entries.extend(_flatten_toc(item))
            continue
        title = getattr(item, "title", None) or getattr(item, "label", None)
        href = getattr(item, "href", None)
        if href:
            entries.append((title or "Untitled", href))
    return entries


def extract_epub(path: str) -> List[Dict[str, Any]]:
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup

    book = epub.read_epub(path)
    toc_entries = _flatten_toc(list(book.toc))
    chapters: List[Dict[str, Any]] = []
    seen_hrefs = set()

    def html_to_text(content: bytes) -> str:
        soup = BeautifulSoup(content, "html.parser")
        return _normalize_text(soup.get_text(" "))

    if toc_entries:
        for index, (title, href) in enumerate(toc_entries, start=1):
            if href in seen_hrefs:
                continue
            item = book.get_item_with_href(href)
            if not item:
                continue
            seen_hrefs.add(href)
            text = html_to_text(item.get_content())
            chapters.append(_chapter_summary(index, title, text))
    else:
        items = list(book.get_items_of_type(ITEM_DOCUMENT))
        for index, item in enumerate(items, start=1):
            title = item.get_name() or f"Chapter {index}"
            text = html_to_text(item.get_content())
            chapters.append(_chapter_summary(index, title, text))

    return chapters


def _flatten_outlines(outlines: List[Any], reader: Any) -> List[Tuple[str, int]]:
    entries: List[Tuple[str, int]] = []
    for item in outlines:
        if isinstance(item, list):
            entries.extend(_flatten_outlines(item, reader))
            continue
        title = getattr(item, "title", None)
        try:
            page_index = reader.get_destination_page_number(item)
        except Exception:
            page_index = None
        if title and page_index is not None:
            entries.append((title, page_index))
    return entries


def extract_pdf(path: str) -> List[Dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    outlines = []
    try:
        outlines = reader.outline
    except Exception:
        try:
            outlines = getattr(reader, "outlines")
        except Exception:
            outlines = []

    chapters: List[Dict[str, Any]] = []
    outline_entries = _flatten_outlines(outlines, reader) if outlines else []
    outline_entries = sorted(outline_entries, key=lambda item: item[1])

    if outline_entries:
        for index, (title, start_page) in enumerate(outline_entries, start=1):
            end_page = (
                outline_entries[index][1]
                if index < len(outline_entries)
                else len(reader.pages)
            )
            text_parts = []
            for page_idx in range(start_page, end_page):
                page_text = reader.pages[page_idx].extract_text() or ""
                text_parts.append(page_text)
            chapters.append(_chapter_summary(index, title, "\n".join(text_parts)))
    else:
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        chapters.append(_chapter_summary(1, "Document", "\n".join(text_parts)))

    return chapters


def extract_book(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".epub":
        chapters = extract_epub(path)
        source_type = "epub"
    elif ext == ".pdf":
        chapters = extract_pdf(path)
        source_type = "pdf"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return {
        "source_path": path,
        "source_type": source_type,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "chapters": chapters,
    }


def write_extract_json(data: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def write_summary_txt(chapters: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w") as f:
        for chapter in chapters:
            line = (
                f"{chapter['index']}. {chapter['title']} | "
                f"words={chapter['word_count']} | preview={chapter['preview']}"
            )
            f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and summarize EPUB/PDF")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="output/book")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    data = extract_book(args.input)
    book_name = os.path.splitext(os.path.basename(args.input))[0]
    output_root = args.output_dir
    output_dir = os.path.join(output_root, book_name)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    summary_path = os.path.join(output_dir, "summary.txt")
    write_summary_txt(data["chapters"], summary_path)
    print(f"Summary saved: {summary_path}")

    if not args.summary_only:
        extract_path = os.path.join(output_dir, "extract.json")
        write_extract_json(data, extract_path)
        print(f"Extract saved: {extract_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
