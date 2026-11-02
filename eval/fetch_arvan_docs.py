"""Download ArvanCloud's public documentation as Markdown, for the evaluation corpus or to upload.

The content is ArvanCloud's and is not committed; this script recreates it under data/arvan-docs/.
    uv run --group eval python eval/fetch_arvan_docs.py           # the evaluation corpus
    uv run --group eval python eval/fetch_arvan_docs.py <URL>...  # chosen pages, into data/pages/
"""

import http.cookiejar
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi
from bs4 import BeautifulSoup
from markdownify import markdownify

BASE = "https://docs.arvancloud.ir"
SECTIONS = {"fa": ["aiaas", "cdn", "object-storage", "accounts"], "en": ["cdn"]}
OUT = Path("data/arvan-docs")
PAGES = Path("data/pages")
DELAY_SECONDS = 1.5  # be polite to the documentation site

opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),  # the site sets a cookie first
    # certifi's CA bundle: some Python builds (uv's on macOS) find no system certificates.
    urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())),
)
opener.addheaders = [("User-Agent", "rag-knowledge-base-eval/0.1 (evaluation corpus)")]


def fetch(url: str) -> str:
    with opener.open(url, timeout=60) as response:
        return str(response.read().decode("utf-8"))


def page_urls(language: str) -> list[str]:
    sitemap = fetch(f"{BASE}/{language}/sitemap.xml")
    urls = re.findall(r"<loc>([^<]+)</loc>", sitemap)
    return [u for u in urls if u.split("/")[4] in SECTIONS[language]]


def to_markdown(page: str, url: str) -> str:
    article = BeautifulSoup(page, "html.parser").find("article")
    if article is None:
        return ""
    for noise in article.select("nav, .theme-doc-toc-mobile, .hash-link, button"):
        noise.decompose()
    body = markdownify(str(article), heading_style="ATX", strip=["img"]).strip()
    return f"<!-- source: {url} -->\n\n{re.sub(r'\n{3,}', '\n\n', body)}\n"


def save_pages(urls: list[str]) -> None:
    """Save each page as Markdown in data/pages/, ready to upload."""
    PAGES.mkdir(parents=True, exist_ok=True)
    for url in urls:
        markdown = to_markdown(fetch(url), url)
        if not markdown:
            print(f"no article found at {url}")
            continue
        path = PAGES / (url.removeprefix(f"{BASE}/").strip("/").replace("/", "__") + ".md")
        path.write_text(markdown, encoding="utf-8")
        print(f"saved {path} ({len(markdown)} chars)")


def main() -> None:
    if sys.argv[1:]:
        save_pages(sys.argv[1:])
        return
    for language in SECTIONS:
        for url in page_urls(language):
            path = OUT / (url.removeprefix(f"{BASE}/").strip("/").replace("/", "__") + ".md")
            if path.exists():
                continue
            try:
                markdown = to_markdown(fetch(url), url)
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"skipped {url}: {exc} (rerun to retry)")
                continue
            if markdown:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(markdown, encoding="utf-8")
                print(f"saved {path.name} ({len(markdown)} chars)")
            time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    main()
