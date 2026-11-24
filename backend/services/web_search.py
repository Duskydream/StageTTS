"""Zero-key stdlib web search helper for proper-noun and IP term verification."""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from typing import Any

_USER_AGENT = (
    "StageTTS/1.0 (+https://github.com/Duskydream/StageTTS; Python urllib)"
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_DDG_LINK_RE = re.compile(
    r'<a[^>]*class=["\'][^"\']*\bresult__a\b[^"\']*["\'][^>]*>(?P<title>[\s\S]*?)</a>',
    re.IGNORECASE,
)
_DDG_SNIPPET_RE = re.compile(
    r'<(?:a|div|span)[^>]*class=["\'][^"\']*\bresult__snippet\b[^"\']*["\'][^>]*>(?P<snippet>[\s\S]*?)</(?:a|div|span)>',
    re.IGNORECASE,
)
_DDG_RESULT_BLOCK_RE = re.compile(
    r'<div[^>]*class=["\'][^"\']*\bresult\b[^"\']*["\'][^>]*>([\s\S]*?)(?=<div[^>]*class=["\'][^"\']*\bresult\b|\Z)',
    re.IGNORECASE,
)


def _strip_html(raw: str) -> str:
    """Remove HTML markup, decode entities, and collapse whitespace."""
    if not raw:
        return ""
    no_tags = _TAG_RE.sub(" ", raw)
    unescaped = html.unescape(no_tags)
    return _WS_RE.sub(" ", unescaped).strip()


def _truncate(text: str, limit: int = 280) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _wiki_lang_code(lang: str | None) -> str | None:
    if not lang:
        return None
    normalized = lang.strip().lower().removeprefix("all_").split("-")[0].split("_")[0]
    if normalized in {"zh", "yue"}:
        return "zh"
    if normalized in {"en", "ja", "ko"}:
        return normalized
    return None


def _http_request(
    url: str,
    *,
    data: bytes | None = None,
    timeout_sec: float = 8.0,
) -> str:
    headers = {"User-Agent": _USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw_bytes = resp.read()
        encoding = resp.headers.get_content_charset() or "utf-8"
        return raw_bytes.decode(encoding, errors="replace")


def _parse_ddg_html(html_text: str, limit: int) -> list[dict[str, str]]:
    """Extract top result titles and snippets from DuckDuckGo HTML markup."""
    if not html_text or limit <= 0:
        return []
    results: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for block in _DDG_RESULT_BLOCK_RE.findall(html_text):
        title_match = _DDG_LINK_RE.search(block)
        if not title_match:
            continue
        title = _truncate(_strip_html(title_match.group("title")), 160)
        snippet_match = _DDG_SNIPPET_RE.search(block)
        snippet = _truncate(_strip_html(snippet_match.group("snippet")), 280) if snippet_match else ""
        if not title and not snippet:
            continue
        key = (title, snippet)
        if key in seen:
            continue
        seen.add(key)
        results.append({"source": "duckduckgo", "title": title, "snippet": snippet})
        if len(results) >= limit:
            return results

    if not results:
        titles = _DDG_LINK_RE.findall(html_text)
        snippets = _DDG_SNIPPET_RE.findall(html_text)
        for idx, raw_title in enumerate(titles):
            title = _truncate(_strip_html(raw_title), 160)
            snippet = _truncate(_strip_html(snippets[idx]), 280) if idx < len(snippets) else ""
            if not title and not snippet:
                continue
            key = (title, snippet)
            if key in seen:
                continue
            seen.add(key)
            results.append({"source": "duckduckgo", "title": title, "snippet": snippet})
            if len(results) >= limit:
                break

    return results


def _search_wikipedia(
    term: str,
    *,
    source_lang: str | None,
    target_lang: str,
    timeout_sec: float,
) -> list[dict[str, str]]:
    """Search Wikipedia for an authoritative entry and target-language interwiki title."""
    wiki_lang = _wiki_lang_code(source_lang) or _wiki_lang_code(target_lang) or "en"
    target_wiki = _wiki_lang_code(target_lang)
    search_params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": term,
            "srlimit": 2,
            "format": "json",
            "utf8": 1,
        }
    )
    search_url = f"https://{wiki_lang}.wikipedia.org/w/api.php?{search_params}"
    payload = json.loads(_http_request(search_url, timeout_sec=timeout_sec))
    hits = payload.get("query", {}).get("search", [])
    if not isinstance(hits, list) or not hits:
        return []

    page_ids = [str(item["pageid"]) for item in hits if isinstance(item, dict) and item.get("pageid")]
    pages_by_id: dict[str, dict[str, Any]] = {}
    if page_ids:
        detail_query: dict[str, Any] = {
            "action": "query",
            "pageids": "|".join(page_ids),
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exchars": 260,
            "format": "json",
            "utf8": 1,
        }
        if target_wiki and target_wiki != wiki_lang:
            detail_query["prop"] = "extracts|langlinks"
            detail_query["lllang"] = target_wiki
            detail_query["lllimit"] = len(page_ids)
        detail_url = f"https://{wiki_lang}.wikipedia.org/w/api.php?{urllib.parse.urlencode(detail_query)}"
        detail_payload = json.loads(_http_request(detail_url, timeout_sec=timeout_sec))
        raw_pages = detail_payload.get("query", {}).get("pages", {})
        if isinstance(raw_pages, dict):
            pages_by_id = {str(k): v for k, v in raw_pages.items() if isinstance(v, dict)}

    results: list[dict[str, str]] = []
    for item in hits:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("pageid") or "")
        page = pages_by_id.get(pid, {})
        title = _truncate(_strip_html(str(page.get("title") or item.get("title") or "")), 160)
        extract = str(page.get("extract") or "")
        snippet = _truncate(_strip_html(extract or str(item.get("snippet") or "")), 280)

        langlinks = page.get("langlinks")
        if isinstance(langlinks, list) and langlinks and isinstance(langlinks[0], dict):
            ll_title = str(langlinks[0].get("*") or langlinks[0].get("title") or "").strip()
            if ll_title:
                snippet = _truncate(f"[{target_wiki}维基对应词条: {ll_title}] {snippet}".strip(), 280)

        if not title and not snippet:
            continue
        results.append(
            {
                "source": f"{wiki_lang}.wikipedia",
                "title": title,
                "snippet": snippet,
            }
        )
    return results


def _search_duckduckgo(
    query: str,
    *,
    limit: int,
    timeout_sec: float,
) -> list[dict[str, str]]:
    """Query DuckDuckGo HTML endpoint with zero API key requirements."""
    if limit <= 0:
        return []
    body = urllib.parse.urlencode({"q": query}).encode("utf-8")
    markup = _http_request("https://html.duckduckgo.com/html/", data=body, timeout_sec=timeout_sec)
    return _parse_ddg_html(markup, limit)


def search_terms(
    terms: list[str],
    *,
    source_lang: str | None = None,
    target_lang: str = "zh",
    max_terms: int = 8,
    max_results_per_term: int = 3,
    timeout_sec: float = 8.0,
) -> dict[str, list[dict[str, str]]]:
    """Look up candidate proper nouns / idioms across Wikipedia and DuckDuckGo safely."""
    cleaned_terms: list[str] = []
    seen_terms: set[str] = set()
    for raw in terms:
        if not isinstance(raw, str):
            continue
        term = _WS_RE.sub(" ", raw).strip()
        if not term:
            continue
        folded = term.casefold()
        if folded in seen_terms:
            continue
        seen_terms.add(folded)
        cleaned_terms.append(term)
        if len(cleaned_terms) >= max(0, max_terms):
            break

    findings: dict[str, list[dict[str, str]]] = {}
    lang_hint = _wiki_lang_code(target_lang)
    ddg_suffix = "中文译名" if lang_hint == "zh" else f"{target_lang} translation"

    for term in cleaned_terms:
        combined: list[dict[str, str]] = []
        try:
            combined.extend(
                _search_wikipedia(
                    term,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    timeout_sec=timeout_sec,
                )
            )
        except Exception:
            pass

        if len(combined) < max_results_per_term:
            try:
                combined.extend(
                    _search_duckduckgo(
                        f"{term} {ddg_suffix}",
                        limit=max_results_per_term,
                        timeout_sec=timeout_sec,
                    )
                )
            except Exception:
                pass

        deduped: list[dict[str, str]] = []
        seen_hits: set[tuple[str, str]] = set()
        for hit in combined:
            title = _truncate(str(hit.get("title") or ""), 160)
            snippet = _truncate(str(hit.get("snippet") or ""), 280)
            if not title and not snippet:
                continue
            key = (title, snippet)
            if key in seen_hits:
                continue
            seen_hits.add(key)
            deduped.append(
                {
                    "source": str(hit.get("source") or "web"),
                    "title": title,
                    "snippet": snippet,
                }
            )
            if len(deduped) >= max_results_per_term:
                break

        if deduped:
            findings[term] = deduped

    return findings
