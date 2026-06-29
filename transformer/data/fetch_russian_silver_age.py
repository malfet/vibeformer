"""Scrape a Russian Silver Age poetry corpus from az.lib.ru (Lib.ru/Классика).

Builds data/russian_silver_age.txt: public-domain verse by Blok, Bryusov,
Annensky, Sologub, Bely, Gippius, Voloshin, Khodasevich, Vyach. Ivanov,
Tsvetaeva and early Mandelstam -- deliberately *excluding* Balmont, so the
file can pretrain a model that is later fine-tuned on tiny_balmont.txt alone.

Pages are cp1251 HTML. For each author we read the index, keep only works
tagged genre "Поэзия", then extract the verse from each work page. Poems are
wrapped in the same ✦ / ✧ markers tiny_balmont.txt uses, so the two corpora
share a format (and, via a shared vocab, a tokenizer).
"""

import html
import os
import re
import sys
import time
import urllib.request

BASE = "http://az.lib.ru"
OUT = os.path.join(os.path.dirname(__file__), "russian_silver_age.txt")

POEM_START = "✦"
POEM_END = "✧"

# (slug, display name). Balmont is intentionally absent.
AUTHORS = [
    ("b/blok_a_a", "Александр Блок"),
    ("b/brjusow_w_j", "Валерий Брюсов"),
    ("a/annenskij_i_f", "Иннокентий Анненский"),
    ("s/sologub_f", "Фёдор Сологуб"),
    ("b/belyj_a", "Андрей Белый"),
    ("g/gippius_z_n", "Зинаида Гиппиус"),
    ("w/woloshin_m_a", "Максимилиан Волошин"),
    ("h/hodasewich_w_f", "Владислав Ходасевич"),
    ("i/iwanow_w_i", "Вячеслав Иванов"),
    ("c/cwetaewa_m_i", "Марина Цветаева"),
    ("m/mandelxshtam_o_e", "Осип Мандельштам"),
    # Gumilyov is absent from Lib.ru/Классика (only on Wikisource), so omitted.
]

# Lines and blocks that signal editorial scaffolding rather than verse.
EDITORIAL_LINE = (
    "Оригинал находится", "Сверка произвед", "OCR", "изд-во", "Печатается по",
    "Книжные полки", "находится по адресу", "Источник:", "Spellcheck",
    "Подготовка текста", "Электронн",
)
EDITORIAL_BLOCK = ("СОСТАВ", "СОДЕРЖАНИЕ", "ОГЛАВЛЕНИЕ")
# Bibliographic header lines that ride along atop "complete works" volumes.
EDITORIAL_LINE_EXTRA = (
    "Полное собрание", "Собрание сочинений", "С.-Пб.", "С.-Петербург",
    '"Наука"', "Том первый", "Том второй", "Том третий", "Библиотека поэта",
)
# Junk codepoints from OCR: C0/C1 controls, NBSP, box-drawing.
JUNK_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\xa0│─]")

# Genre tokens az.lib.ru prints in each index entry's trailing <small>.
GENRES = [
    "Поэзия", "Проза", "Драматургия", "Переводы", "Критика", "Публицистика",
    "Мемуары", "Философия", "Религия", "Публ", "Сказки", "Детская", "История",
    "Эпистолярий", "Юмор и сатира",
]


def fetch(url, retries=3):
    """GET a cp1251 page and return it decoded as UTF-8 text."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read()
            return raw.decode("cp1251", errors="replace")
        except Exception as e:  # noqa: BLE001 - best-effort scraper
            if attempt == retries - 1:
                print(f"    ! fetch failed {url}: {e}", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))
    return None


def index_poetry_works(index_html):
    """Return [(href, title)] for index entries whose genre is 'Поэзия'."""
    # Split the index into one chunk per work entry.
    entries = re.split(r"<DT><li>", index_html, flags=re.I)
    works = []
    for e in entries:
        m = re.search(r'<A HREF=(text_[^>\s]+\.shtml)>\s*<b>(.*?)</b>', e, re.I | re.S)
        if not m:
            continue
        href, title = m.group(1), html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        # Genre is the first known genre token in the entry's trailing markup.
        genre = next((g for g in GENRES if re.search(r"[\s>]" + re.escape(g) + r"[\s<]", e)), None)
        if genre == "Поэзия":
            works.append((href, title))
    return works


def extract_poems(work_html):
    """Pull verse out of a work page, returned as a list of poem strings.

    Handles both az.lib.ru layouts: small single works (verse in <dd> lines)
    and big collections (plain-text poems split by <A name=NN> anchors with a
    table-of-contents of <a href=#NN> links on top).
    """
    # Isolate the work body. The start/end markers live *inside* HTML comments,
    # so slice past the full opening comment and before the closing one -- not
    # mid-comment, or fragments like "Собственно произведение --->" leak through.
    m = work_html.find("Собственно произведение")
    if m == -1:
        return []
    op_end = work_html.find("-->", m)
    body = work_html[op_end + 3 if op_end != -1 else m:]
    end = -1
    for end_marker in ("Блок описания произведения", "Блочек голосования",
                       "sape.ru request"):
        end = body.find(end_marker)
        if end != -1:
            break
    if end != -1:
        comment_start = body.rfind("<!--", 0, end)
        body = body[:comment_start if comment_start != -1 else end]

    # Drop table-of-contents lines (internal #anchor links) outright.
    body = "\n".join(l for l in body.splitlines() if not re.search(r"<a href=#", l, re.I))

    # Mark poem boundaries at each <A name=...> anchor, drop any stray HTML
    # comments, then strip all remaining tags.
    body = re.sub(r"<a\s+name=[^>]*>", "\x00", body, flags=re.I)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"<[^>]+>", "\n", body)
    body = html.unescape(body).replace("\xa0", " ")

    chunks = body.split("\x00")
    poems = []
    for chunk in chunks:
        lines = []
        for l in chunk.splitlines():
            l = JUNK_CHARS.sub("", l)
            l = re.sub(r"[ \t]+", " ", l).rstrip()
            # Drop separator rules (----, ====, ****) and editorial one-liners.
            if re.fullmatch(r"[\s\-=*_~.]{4,}", l):
                continue
            if any(m in l for m in EDITORIAL_LINE + EDITORIAL_LINE_EXTRA):
                continue
            lines.append(l)
        # Collapse 3+ blank lines to 2; trim leading/trailing blanks.
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n ")
        # Skip title-page / table-of-contents chunks wholesale.
        if any(m in text for m in EDITORIAL_BLOCK):
            continue
        # Keep only chunks that are real verse: enough Cyrillic, a few lines.
        cyr = sum("а" <= c.lower() <= "я" or c in "ёЁ" for c in text)
        if cyr >= 80 and text.count("\n") >= 2:
            poems.append(text)
    return poems


def scrape_author(slug, name):
    index = fetch(f"{BASE}/{slug}/")
    if index is None:
        return "", 0
    works = index_poetry_works(index)
    print(f"  {name}: {len(works)} poetry works")
    out = []
    n_poems = 0
    for href, title in works:
        page = fetch(f"{BASE}/{slug}/{href}")
        if page is None:
            continue
        poems = extract_poems(page)
        n_poems += len(poems)
        for p in poems:
            out.append(f"{POEM_START}\n{p}\n{POEM_END}\n")
        time.sleep(0.3)
    text = "\n".join(out)
    print(f"    -> {n_poems} poems, {len(text):,} chars")
    return text, n_poems


def main():
    only = sys.argv[1:] or None  # optional author-slug filter for testing
    parts = []
    summary = []
    for slug, name in AUTHORS:
        if only and not any(o in slug for o in only):
            continue
        text, n = scrape_author(slug, name)
        if text:
            parts.append(text)
            summary.append((name, n, len(text)))
    corpus = "\n".join(parts)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(corpus)
    print("\n=== summary ===")
    for name, n, c in summary:
        print(f"  {name:24s} {n:5d} poems  {c:>10,} chars")
    print(f"  {'TOTAL':24s} {sum(s[1] for s in summary):5d} poems  {len(corpus):>10,} chars")
    print(f"  written to {OUT}")


if __name__ == "__main__":
    main()
