"""Gate for dataset acquisition — the no-network paths of ensure_corpus.

We never hit the network in tests: existing files return immediately, and an
unknown missing path raises a clear error rather than silently downloading.
"""

from __future__ import annotations

import pytest

from thcm.data.fetch import ensure_corpus


def test_existing_corpus_returns_path(tmp_path) -> None:
    p = tmp_path / "mydata.txt"
    p.write_bytes(b"hello bytes")
    assert ensure_corpus(str(p)) == str(p)            # no download attempted


def test_unknown_missing_corpus_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="corpus not found"):
        ensure_corpus(str(tmp_path / "not_a_dataset.txt"))


def test_existing_enwik8_is_not_redownloaded(tmp_path) -> None:
    # A present enwik8 must short-circuit before any network access.
    p = tmp_path / "enwik8"
    p.write_bytes(b"\x00" * 32)
    assert ensure_corpus(str(p)) == str(p)
