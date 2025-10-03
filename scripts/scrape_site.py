import sys
from collections import deque
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
import trafilatura
from bs4 import BeautifulSoup


def sanitize_filename(url: str) -> str:
    p = urlparse(url)
    host = p.netloc.replace(":", "_")
    slug = p.path.strip("/").replace("/", "_") or "index"
    qs = ("_" + p.query.replace("/", "_").replace("=", "-").replace("&", "+")) if p.query else ""
    return f"{host}_{slug}{qs}.txt"


def save_page_text(url: str, out_dir: Path, timeout: int = 20) -> Path | None:
    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
    except TypeError:
        # older trafilatura versions don't support timeout
        downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
    ) or ""

    if not text.strip():
        # fallback to raw HTML if structured text fails
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (RAG bot)"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # strip scripts/styles
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text("\n")
        except Exception:
            return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / sanitize_filename(url)
    try:
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved: {out_path}")
    except Exception:
        return None
    return out_path


def is_same_domain(url: str, base_netloc: str) -> bool:
    return urlparse(url).netloc == base_netloc


def is_valid_link(href: str) -> bool:
    if not href:
        return False
    href = href.strip()
    if href.startswith("mailto:") or href.startswith("tel:"):
        return False
    if any(href.lower().endswith(ext) for ext in [
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".mp4", ".mp3", ".zip", ".pdf", ".doc", ".docx",
        ".xls", ".xlsx", ".ppt", ".pptx",
    ]):
        return False
    return True


def crawl(start_url: str, out_dir: Path, max_pages: int = 100, timeout: int = 20):
    parsed = urlparse(start_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    base_netloc = parsed.netloc

    q = deque([start_url])
    seen = set()
    saved = 0

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (RAG bot)"})

    while q and saved < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)

        # Save page text
        if save_page_text(url, out_dir, timeout=timeout):
            saved += 1

        # Fetch page to extract links
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
        except Exception:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not is_valid_link(href):
                continue
            full = urljoin(url, href)
            # same-domain only
            if is_same_domain(full, base_netloc) and full not in seen:
                q.append(full)

    print(f"Crawled {saved} pages from {base}")


if __name__ == "__main__":
    # Simple CLI:
    # 1) Single page mode: python scripts/scrape_site.py https://koremobiles.com/
    # 2) Crawl mode (domain): python scripts/scrape_site.py crawl https://koremobiles.com/ 150
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "data" / "websites"

    if len(sys.argv) == 1:
        print("Usage:\n  python scripts/scrape_site.py <URL1> <URL2> ...\n  python scripts/scrape_site.py crawl <START_URL> [MAX_PAGES]")
        sys.exit(1)

    if sys.argv[1].lower() == "crawl":
        if len(sys.argv) < 3:
            print("Usage: python scripts/scrape_site.py crawl <START_URL> [MAX_PAGES]")
            sys.exit(1)
        start = sys.argv[2]
        max_pages = int(sys.argv[3]) if len(sys.argv) >= 4 else 150
        crawl(start, out_dir, max_pages=max_pages)
    else:
        for url in sys.argv[1:]:
            save_page_text(url, out_dir)
