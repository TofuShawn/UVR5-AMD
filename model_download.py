"""Weight downloads, and nothing else this app fetches for itself.

The fork may now do network I/O for **user-initiated model downloads** and only for
that: the update channel stays deleted, `UVR_MODEL_ROOT` stays read-only, and the
destination is the per-user config directory (`settings.config_dir()/models`), which
the catalog scans in addition to the read-only root.

Three properties, each measured rather than assumed (2026-11-09):

* The weights release 302s to `release-assets.githubusercontent.com`. An allowlist of
  `github.com` / `raw.githubusercontent.com` / `dl.fbaipublicfiles.com` alone refuses
  every GitHub download at the first hop, so the asset hosts are part of the channel.
* The same URL honours `Range: bytes=0-` with **206**, so a retry resumes instead of
  restarting a 300 MB file — which is what "download never moves" looks like.
* No checksum is published. The only integrity claim made here is `Content-Length`
  against bytes written; the catalog's tail hash is computed after the file lands.

Failure is loud and leaves the `.part` behind with the byte count in the message, so a
stalled fetch reads as "stopped at 65,536 bytes" rather than as a frozen bar.
"""
from __future__ import annotations

import http.client
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

__all__ = ["ALLOWED_HOSTS", "WEIGHTS_BASE", "DownloadCancelled", "DownloadFailed",
           "DownloadRefused", "check_url", "default_models_dir", "download",
           "weight_url"]

#: The weights channel (`NORMAL_REPO` in `gui_data/constants.py`), plus the two hosts
#: its release assets redirect through. HTTPS only; a redirect that leaves this set is
#: refused before the next request is made.
ALLOWED_HOSTS = frozenset({
    "github.com",
    "raw.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
    "dl.fbaipublicfiles.com",
})

WEIGHTS_BASE = ("https://github.com/TRvlvr/model_repo/releases/download/"
                "all_public_uvr_models/")

WEIGHT_SUFFIXES = frozenset({".pth", ".pt", ".th", ".ckpt", ".bin", ".safetensors"})

CHUNK = 64 * 1024


class DownloadRefused(Exception):
    """Refused by policy, before any byte was requested: URL, host, or destination."""


class DownloadFailed(Exception):
    """The transfer did not complete. The `.part` file and its size are kept."""

    def __init__(self, message: str, *, received: int = 0):
        super().__init__(message)
        self.received = received


class DownloadCancelled(DownloadFailed):
    """The user stopped it; the `.part` file is kept for a resume."""


def default_models_dir() -> Path:
    """Per-user, next to `settings.json`; never `UVR_MODEL_ROOT`, which is read-only."""
    from settings import config_dir

    return config_dir() / "models"


def weight_url(name: str) -> str:
    """The public release URL for a bare weight file name.

    A name is a leaf, not a path: `catalog` supplies real names, but a manifest edit
    must not be able to turn one into `../../startup.pth` inside a config directory.
    """
    if (not name or name in {".", ".."} or name.startswith(".")
            or "/" in name or "\\" in name):
        raise DownloadRefused(f"not a bare file name: {name!r}")
    if Path(name).suffix.lower() not in WEIGHT_SUFFIXES:
        raise DownloadRefused(f"not a weight suffix: {name!r}")
    return WEIGHTS_BASE + urllib.parse.quote(name)


def check_url(url: str) -> urllib.parse.SplitResult:
    """HTTPS, an allowlisted host, the default port, no userinfo."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise DownloadRefused(f"not https: {url}")
    if parts.username or parts.password:
        raise DownloadRefused(f"credentials in a model URL: {url}")
    try:
        port = parts.port
    except ValueError as exc:
        raise DownloadRefused(f"unreadable port: {url}") from exc
    if port not in (None, 443):
        raise DownloadRefused(f"non-default port: {url}")
    if parts.hostname not in ALLOWED_HOSTS:
        raise DownloadRefused(f"host is not on the allowlist: {parts.hostname}")
    return parts


class _AllowlistRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse at the hop, not after the off-list host has already served bytes."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_fetcher(request: urllib.request.Request, timeout: float):
    opener = urllib.request.build_opener(_AllowlistRedirect())
    return opener.open(request, timeout=timeout)


def download(name: str, dest_dir: Path | str | None = None, *,
             fetcher=None, on_progress=None, cancel=None,
             timeout: float = 30.0, resume: bool = True) -> Path:
    """Fetch one weight into `dest_dir` (default `default_models_dir()`).

    `fetcher(request, timeout)` returns a response with `read`, `headers`, `status`,
    `geturl` and `close`; tests inject it, production uses urllib. `on_progress(got,
    total)` is called per chunk (`total` is None when the server omits Content-Length).
    `cancel()` is polled between chunks.
    """
    url = weight_url(name)
    dest = Path(dest_dir) if dest_dir is not None else default_models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / name
    if final.is_file():
        return final
    part = dest / f"{name}.part"
    received = part.stat().st_size if (resume and part.is_file()) else 0
    if not resume:
        part.unlink(missing_ok=True)

    fetch = fetcher or _default_fetcher
    restarted = False
    while True:
        headers = {"Range": f"bytes={received}-"} if received else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            response = fetch(request, timeout)
        except (TimeoutError, socket.timeout, urllib.error.URLError, OSError) as exc:
            raise DownloadFailed(f"request failed after {received} bytes: {exc}",
                                 received=received) from exc
        try:
            check_url(response.geturl())
            status = getattr(response, "status", 200)
            if received and status != 206:
                if restarted:
                    raise DownloadFailed(
                        f"server answered {status} to a resumed request after "
                        f"{received} bytes", received=received)
                restarted = True
                received = 0
                part.unlink(missing_ok=True)
                continue
            length = response.headers.get("Content-Length")
            total = int(length) + received if length not in (None, "") else None
            with open(part, "ab" if received else "wb") as handle:
                while True:
                    if cancel is not None and cancel():
                        raise DownloadCancelled(
                            f"cancelled after {received} bytes of {total or '?'}",
                            received=received)
                    try:
                        chunk = response.read(CHUNK)
                    except (TimeoutError, socket.timeout, OSError,
                            http.client.HTTPException) as exc:
                        raise DownloadFailed(
                            f"stalled after {received} bytes of {total or '?'}: {exc}",
                            received=received) from exc
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                    if on_progress is not None:
                        on_progress(received, total)
        finally:
            response.close()
        if total is not None and received != total:
            raise DownloadFailed(
                f"short body: {received} of {total} bytes for {name}",
                received=received)
        os.replace(part, final)
        return final
