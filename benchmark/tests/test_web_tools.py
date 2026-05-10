import types
import pytest
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import benchmark...` works
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.eval.tools import searcher, browser


def _install_missing_module(monkeypatch, name: str):
    # Install a module without required attributes to simulate missing dependency
    mod = types.ModuleType(name)
    monkeypatch.setitem(__import__('sys').modules, name, mod)
    return mod


def _install_requests_stub(monkeypatch, html: str, status: int = 200, url: str = "https://example.com/search"):
    mod = types.ModuleType("requests")

    class _Resp:
        def __init__(self, text: str, status_code: int, url: str):
            self.text = text
            self.status_code = status_code
            self.url = url

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

    def _get(*a, **k):
        return _Resp(html, status, url)

    mod.get = _get
    monkeypatch.setitem(__import__('sys').modules, 'requests', mod)
    return mod


def _install_bs4_search_stub(monkeypatch):
    """Install a lightweight BeautifulSoup stub sufficient for searcher()."""
    mod = types.ModuleType('bs4')

    class _Tag:
        def __init__(self, text: str = "", attrs: dict | None = None):
            self._text = text
            self._attrs = attrs or {}

        def get_text(self, sep: str = " ", strip: bool = False):
            return self._text

        def get(self, name: str, default=None):
            return self._attrs.get(name, default)

        def __getitem__(self, key: str):
            return self._attrs[key]

    class _Container:
        def __init__(self, title: str, url: str, snippet: str):
            self._title = title
            self._url = url
            self._snippet = snippet

        def select_one(self, selector: str):
            if selector == 'a':
                return _Tag(attrs={'href': self._url})
            if selector == 'h3':
                return _Tag(text=self._title)
            if selector in ('div.VwiC3b', 'div.IsZvec', 'span.aCOpRe'):
                return _Tag(text=self._snippet)
            return None

    class BeautifulSoup:  # noqa: N801 - match import name
        def __init__(self, html: str, parser: str):
            self._html = html

        def select(self, selector: str):
            # Always return two deterministic containers regardless of HTML
            if selector in ('div.tF2Cxc', 'div.g'):
                return [
                    _Container("Result One", "https://r1.example", "Snippet 1"),
                    _Container("Result Two", "https://r2.example", "Snippet 2"),
                ]
            return []

    mod.BeautifulSoup = BeautifulSoup
    monkeypatch.setitem(__import__('sys').modules, 'bs4', mod)
    return mod


def _install_bs4_browser_stub(monkeypatch):
    """Install a lightweight BeautifulSoup stub sufficient for browser()."""
    mod = types.ModuleType('bs4')

    class _Title:
        def __init__(self, text: str):
            self.string = text

    class _ATag:
        def __init__(self, text: str, href: str):
            self._text = text
            self._attrs = {'href': href}

        def get_text(self, sep: str = " ", strip: bool = False):
            return self._text

        def __getitem__(self, key: str):
            return self._attrs[key]

    class _Decomp:
        def decompose(self):
            pass

    class BeautifulSoup:  # noqa: N801
        def __init__(self, html: str, parser: str):
            self._html = html
            self.title = _Title("Stub Page Title")

        def __call__(self, names):
            # Simulate soup(["script", "style", ...])
            return [_Decomp() for _ in names]

        def get_text(self, sep: str = "\n", strip: bool = True):
            return "Stub content body"

        def find_all(self, name: str, href: bool = False):
            if name == 'a':
                return [
                    _ATag("Link 1", "/path1"),
                    _ATag("Link 2", "https://ext.example/path2"),
                ]
            return []

    mod.BeautifulSoup = BeautifulSoup
    monkeypatch.setitem(__import__('sys').modules, 'bs4', mod)
    return mod


def test_searcher_dependency_missing(monkeypatch):
    # Ensure bs4 BeautifulSoup import fails gracefully
    _install_missing_module(monkeypatch, 'bs4')
    res = searcher("triton kernels")
    assert res == []


def test_searcher_parses_results(monkeypatch):
    _install_requests_stub(monkeypatch, html="<html>ignored</html>")
    _install_bs4_search_stub(monkeypatch)
    res = searcher("triton add")
    assert isinstance(res, list) and len(res) >= 2
    assert res[0]["title"] == "Result One"
    assert res[0]["url"].startswith("https://")
    assert "snippet" in res[0]


def test_browser_dependency_missing(monkeypatch):
    _install_missing_module(monkeypatch, 'bs4')
    out = browser("https://example.com")
    assert out.get("status") == "error"
    assert "missing dependency" in out.get("error", "")


def test_browser_parses_content(monkeypatch):
    _install_requests_stub(monkeypatch, html="<html>ignored</html>", url="https://example.com/page")
    _install_bs4_browser_stub(monkeypatch)
    out = browser("https://example.com/page")
    assert out["status"] == "ok"
    assert out["title"] == "Stub Page Title"
    assert out["url"].startswith("https://example.com")
    assert isinstance(out["links"], list) and len(out["links"]) >= 2
    assert all("url" in l and "text" in l for l in out["links"]) 
