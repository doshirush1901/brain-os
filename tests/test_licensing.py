"""License key format validation."""

from __future__ import annotations

import pytest
from brain_os.licensing.activate import activate_with_key


def test_activate_rejects_bad_key_format() -> None:
    with pytest.raises(ValueError, match="Invalid key format"):
        activate_with_key("not-a-valid-key")


def test_activate_rejects_short_bos_prefix() -> None:
    with pytest.raises(ValueError, match="Invalid key format"):
        activate_with_key("bos_live_short")
