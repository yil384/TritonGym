

from __future__ import annotations
import json
import tempfile
from importlib import util
from pathlib import Path
from typing import Tuple

import torch
from urllib.parse import quote_plus, urlparse, parse_qs, urljoin

from .runner import GenericEvaluator, TaskSpec, EvalConfig

BENCHMARK_JSON = Path("benchmark/data_v2/benchmark.json")


def _load_spec(op: str) -> Tuple[str, list]:
    if not BENCHMARK_JSON.is_file():
        raise FileNotFoundError(f"Spec file not found: {BENCHMARK_JSON}")
    spec = json.loads(BENCHMARK_JSON.read_text())
    if op not in spec:
        raise KeyError(f"Operator '{op}' not found in {BENCHMARK_JSON}")
    entry = spec[op]
    torch_src = entry.get("torch_operator") or entry.get("torch") or ""
    input_shapes = entry.get("input_shapes") or []
    return torch_src, input_shapes


def _make_evaluator(op: str) -> GenericEvaluator:
    torch_src, input_shapes = _load_spec(op)

    def _correctness(out, ref):
        with torch.no_grad():
            diff = (out - ref).abs()
            denom = ref.abs().clamp_min(1e-8)
            rel = diff / denom
            rel = torch.where(torch.isfinite(rel), rel, torch.zeros_like(rel))
            max_abs = torch.nan_to_num(diff, nan=0.0, posinf=float('inf'), neginf=float('inf')).max().item()
            max_rel = rel.max().item()
        return {"max_abs": max_abs, "max_rel": max_rel}

    spec = TaskSpec(
        name=op,
        reference_source=torch_src,
        correctness_fn=_correctness,
        input_shapes=input_shapes,
        flops_fn=None,
    )
    cfg = EvalConfig(warmup=5, repeat=10, shape_index=0)
    return GenericEvaluator(cfg, spec)


def evaluator(src: str, evaluator_obj: GenericEvaluator) -> float:
    """Compile and run the generated function triton_<op>; return avg latency (ms).

    Returns NaN if compilation/runtime fails.
    """
    try:
        res = evaluator_obj.evaluate_code(src)
        return float("nan") if (not res.compile_success or res.runtime_ms is None) else float(res.runtime_ms)
    except Exception:
        return float("nan")


def verifier(src: str, evaluator_obj: GenericEvaluator) -> str:
    """Check correctness of generated function triton_<op> vs. torch reference.

    Uses simple thresholds suitable for fp16/fp32.
    """
    try:
        res = evaluator_obj.evaluate_code(src)
        if not res.compile_success:
            return "compile failed"
        if not isinstance(res.metrics, dict):
            return "no metrics returned"
        if "error" in res.metrics:
            return f"runtime error: {res.metrics['error']}"
        max_abs = res.metrics.get("max_abs", float("inf"))
        if max_abs <= 1e-2:
            return ""
        else:
            return f"tolerance exceeded (max_abs={max_abs})"
    except Exception as e:
        return f"exception: {e}"

def compiler(src: str, evaluator_obj: GenericEvaluator) -> str:
    """Attempt to compile the generated code and locate triton_<op> function.

    Returns empty string on success, else the error message.
    """
    expected_name = f"triton_{evaluator_obj.spec.name}"
    try:
        tmpdir = Path(tempfile.mkdtemp())
        code_path = tmpdir / "gen_impl.py"
        code_path.write_text(src)
        m_spec = util.spec_from_file_location("gen_impl", code_path)
        module = util.module_from_spec(m_spec)
        assert m_spec and m_spec.loader
        m_spec.loader.exec_module(module)  # type: ignore
        if not hasattr(module, expected_name):
            return f"symbol {expected_name} missing"
        fn = getattr(module, expected_name)
        if isinstance(fn, type) or not callable(fn):
            return "expected a callable function"
        return ""
    except Exception as e:
        return str(e)

def searcher(key: str):
    """Search Google for the given key and return parsed results.

    Returns a list of dicts: {title, url, snippet}. Empty list on failure.
    """
    try:
        # Lazy import to avoid hard dependency if not used
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return []

    try:
        q = quote_plus(key)
        url = f"https://www.google.com/search?q={q}&num=10&hl=en"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        # Prefer new-style result containers
        containers = soup.select("div.tF2Cxc")
        if not containers:
            # Fallback to generic result blocks
            containers = soup.select("div.g")

        def _clean_href(href: str) -> str:
            if not href:
                return ""
            if href.startswith("/url?"):
                parsed = urlparse(href)
                qs = parse_qs(parsed.query)
                real = qs.get("q", [""])[0]
                return real
            return href

        for c in containers:
            a = c.select_one("a")
            h3 = c.select_one("h3")
            if not a or not h3:
                continue
            href = _clean_href(a.get("href", ""))
            title = h3.get_text(strip=True)
            # Snippet heuristics across different DOMs
            snippet_el = (
                c.select_one("div.VwiC3b")
                or c.select_one("div.IsZvec")
                or c.select_one("span.aCOpRe")
            )
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            if href and title:
                results.append({"title": title, "url": href, "snippet": snippet})
        return results
    except Exception:
        return []


def browser(url: str):
    """Fetch a URL and return parsed content using BeautifulSoup.

    Returns a dict with keys: status, title, html, text, links.
    On error, returns {status: 'error', error: <message>}.
    """
    try:
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as e:
        return {"status": "error", "error": f"missing dependency: {e}"}

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Title
        title = (soup.title.string or "").strip() if soup.title else ""

        # Text (strip scripts/styles)
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)

        # Links (absolute)
        links = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = urljoin(resp.url, a["href"])  # resolve relative links
            if href in seen:
                continue
            seen.add(href)
            label = a.get_text(" ", strip=True)
            links.append({"text": label, "url": href})
            if len(links) >= 25:
                break

        # Optionally limit very large payloads
        html = resp.text
        if len(html) > 500_000:
            html = html[:500_000]
        if len(text) > 200_000:
            text = text[:200_000]

        return {
            "status": "ok",
            "title": title,
            "html": html,
            "text": text,
            "links": links,
            "url": resp.url,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}