"""The downloader the Models page will call: policy first, then bytes.

The bilibili thread under a UVR tutorial has one poster answering "下載一直下不動呢"
(2026-06), and the packaged build's own answer to "too many models" is "the settings
page can download a pile". So the fork's downloader is judged by the failure it
replaces, not by the happy path: a fetch that hangs forever, a half-written weight
that the catalog then refuses, a redirect that walks off the allowlist.

Three measurements taken before this file was written, all of them shaping assertions:

* `curl -I` on the weights release (2026-11-09) answers **302** with
  `Location: https://release-assets.githubusercontent.com/github-production-release-asset/...`.
  The allowlist the design recorded (`github.com`, `raw.githubusercontent.com`,
  `dl.fbaipublicfiles.com`) therefore refuses **every** GitHub weight download at its
  first hop. `release-assets.githubusercontent.com` (and the older
  `objects.githubusercontent.com`) are part of the channel or nothing is downloadable.
* The same URL with `Range: bytes=0-2097151` answers **206** and 2 MiB in 0.5 s, so
  resume is a real option and not a wish.
* No checksum is published for these weights. Verification is `Content-Length`
  against bytes written, then the catalog's own tail hash after the file lands. Claiming
  integrity we cannot check would be the lie this module cannot afford.

Nothing here touches the network: `fetcher` is injected exactly the way
`model_refresh.py` injects its opener, and the default opener is exercised only by the
allowlist tests, which pass URLs to `check_url` rather than to a socket.
"""
from __future__ import annotations

import io
import socket

import pytest

import model_download
from model_download import (ALLOWED_HOSTS, DownloadCancelled, DownloadFailed,
                            DownloadRefused, check_url, download, weight_url)

WEIGHTS_BASE = ("https://github.com/TRvlvr/model_repo/releases/download/"
                "all_public_uvr_models/")


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #

class FakeResponse:
    """The five things `download` is allowed to ask of a response."""

    def __init__(self, data: bytes, *, url: str | None = None, status: int = 200,
                 length: int | None = -1, fail_after: int | None = None):
        self._buf = io.BytesIO(data)
        self._url = url or WEIGHTS_BASE + "1_HP-UVR.pth"
        self.status = status
        self.headers = {"Content-Length": str(len(data)) if length == -1 else
                        (None if length is None else str(length))}
        self._fail_after = fail_after
        self._served = 0
        self.closed = False

    def read(self, n: int = -1) -> bytes:
        if self._fail_after is not None and self._served >= self._fail_after:
            raise socket.timeout("timed out")
        chunk = self._buf.read(n)
        self._served += len(chunk)
        return chunk

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        self.closed = True


class Fetcher:
    """Records each request (`Range` header included) and hands out canned responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append({"url": request.full_url,
                              "range": request.get_header("Range"),
                              "timeout": timeout})
        return self.responses.pop(0)


# --------------------------------------------------------------------------- #
# 1. names and URLs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [
    "", ".", "..", "../evil.pth", "sub/evil.pth", "sub\\evil.pth",
    "/abs/evil.pth", "evil.exe", "evil.pth.txt", ".hidden.pth",
])
def test_only_a_bare_weight_file_name_can_become_a_url(bad):
    """A name is a leaf, not a path, and its suffix is one the engine loads.

    `catalog` names come from the model tables, but the downloader is also reachable
    from a manifest and a user-typed name must never become `../../startup.pth`
    inside a per-user config directory.
    """
    with pytest.raises(DownloadRefused):
        weight_url(bad)


def test_a_weight_url_is_the_public_release_channel():
    assert weight_url("1_HP-UVR.pth") == WEIGHTS_BASE + "1_HP-UVR.pth"
    assert weight_url("BS-Roformer-Viperx-1297.ckpt") == (
        WEIGHTS_BASE + "BS-Roformer-Viperx-1297.ckpt")
    # A space is legal in a file name and must arrive percent-encoded, not raw.
    assert weight_url("my model.pth") == WEIGHTS_BASE + "my%20model.pth"


# --------------------------------------------------------------------------- #
# 2. the allowlist, including the hop nobody wrote down
# --------------------------------------------------------------------------- #

def test_the_release_asset_redirect_target_is_on_the_allowlist():
    """Measured 2026-11-09: `github.com/.../releases/download/...` 302s to
    `release-assets.githubusercontent.com`. Without this host every real download is
    refused after zero bytes, which is indistinguishable to a user from a hang."""
    assert "release-assets.githubusercontent.com" in ALLOWED_HOSTS
    assert "objects.githubusercontent.com" in ALLOWED_HOSTS
    checked = check_url("https://release-assets.githubusercontent.com/github-production-"
                        "release-asset/514182217/x?sp=r&sig=y")
    assert checked.hostname == "release-assets.githubusercontent.com"


@pytest.mark.parametrize("url", [
    "http://github.com/x.pth",                        # not https
    "https://evil.example.com/x.pth",                 # not on the list
    "https://github.com.evil.example/x.pth",          # suffix trick
    "https://github.com@evil.example/x.pth",          # userinfo trick
    "https://github.com:8443/x.pth",                  # non-default port
])
def test_an_off_list_url_is_refused_before_any_byte_is_requested(url):
    with pytest.raises(DownloadRefused):
        check_url(url)


def test_a_redirect_off_the_allowlist_is_refused_by_the_handler():
    """`check_url` alone would only see the *final* URL, after the off-list host had
    already served bytes. The handler refuses at the hop, and the redirect is also
    re-checked after the fact because the fetcher is injectable."""
    handler = model_download._AllowlistRedirect()
    with pytest.raises(DownloadRefused):
        handler.redirect_request(None, None, 302, "Found", {},
                                 "https://evil.example/x.pth")


# --------------------------------------------------------------------------- #
# 3. a download that lands
# --------------------------------------------------------------------------- #

def test_a_download_writes_the_part_file_first_and_renames_atomically(tmp_path):
    data = b"weights" * 5000
    fetcher = Fetcher(FakeResponse(data))
    seen = []
    final = download("1_HP-UVR.pth", tmp_path, fetcher=fetcher,
                     on_progress=lambda got, total: seen.append((got, total)))

    assert final == tmp_path / "1_HP-UVR.pth"
    assert final.read_bytes() == data
    assert not (tmp_path / "1_HP-UVR.pth.part").exists()
    assert seen[-1] == (len(data), len(data))
    assert [got for got, _ in seen] == sorted(got for got, _ in seen)


def test_an_already_installed_model_is_not_fetched_again(tmp_path):
    (tmp_path / "1_HP-UVR.pth").write_bytes(b"already here")
    fetcher = Fetcher()          # pops raise IndexError if called
    assert download("1_HP-UVR.pth", tmp_path, fetcher=fetcher).read_bytes() == b"already here"
    assert fetcher.requests == []


# --------------------------------------------------------------------------- #
# 4. the failure the thread reported: it stops moving
# --------------------------------------------------------------------------- #

def test_a_stalled_connection_fails_with_the_byte_count_instead_of_hanging(tmp_path):
    """"下載一直下不動" is a socket timeout with no status line. The user's evidence has
    to be the number of bytes that did arrive, and the `.part` has to survive it --
    otherwise every retry starts from zero on a 300 MB file and looks stalled too."""
    data = b"w" * 200_000
    fetcher = Fetcher(FakeResponse(data, fail_after=65_536))
    with pytest.raises(DownloadFailed) as excinfo:
        download("1_HP-UVR.pth", tmp_path, fetcher=fetcher, timeout=7.5)

    unsent = tmp_path / "1_HP-UVR.pth.part"
    assert unsent.stat().st_size == 65_536
    assert not (tmp_path / "1_HP-UVR.pth").exists()
    assert fetcher.requests[0]["timeout"] == 7.5
    assert "65536" in str(excinfo.value)


def test_the_retry_resumes_at_the_part_file_with_a_range_request(tmp_path):
    data = b"0123456789" * 20_000          # 200 000 bytes
    half = len(data) // 2
    (tmp_path / "1_HP-UVR.pth.part").write_bytes(data[:half])
    fetcher = Fetcher(FakeResponse(data[half:], status=206))

    final = download("1_HP-UVR.pth", tmp_path, fetcher=fetcher)
    assert final.read_bytes() == data
    assert fetcher.requests[0]["range"] == f"bytes={half}-"


def test_a_server_that_ignores_range_restarts_instead_of_appending(tmp_path):
    """A 200 to a ranged request means the server is sending the whole file; appending
    it to a partial would corrupt the weight and the catalog would refuse it later."""
    data = b"x" * 10_000
    (tmp_path / "1_HP-UVR.pth.part").write_bytes(b"old-garbage")
    fetcher = Fetcher(FakeResponse(data, status=200), FakeResponse(data, status=200))

    final = download("1_HP-UVR.pth", tmp_path, fetcher=fetcher)
    assert final.read_bytes() == data


# --------------------------------------------------------------------------- #
# 5. cancel and truncation
# --------------------------------------------------------------------------- #

def test_cancel_leaves_the_part_file_and_no_model(tmp_path):
    data = b"y" * 100_000
    fetcher = Fetcher(FakeResponse(data))
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(DownloadCancelled):
        download("1_HP-UVR.pth", tmp_path, fetcher=fetcher, cancel=cancel)

    assert not (tmp_path / "1_HP-UVR.pth").exists()
    assert (tmp_path / "1_HP-UVR.pth.part").stat().st_size < len(data)


def test_a_short_body_is_a_failure_not_a_model(tmp_path):
    data = b"z" * 60
    fetcher = Fetcher(FakeResponse(data, length=100))
    with pytest.raises(DownloadFailed):
        download("1_HP-UVR.pth", tmp_path, fetcher=fetcher)
    assert not (tmp_path / "1_HP-UVR.pth").exists()
    assert (tmp_path / "1_HP-UVR.pth.part").stat().st_size == 60
