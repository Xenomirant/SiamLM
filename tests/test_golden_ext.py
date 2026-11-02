"""SiamLM sends the requests the original harness sent, for every feature added after the first golden file
(tests/golden/extended.json, recorded from the original code with the same scripts; see golden_ext.py)."""
import json

import pytest

from golden_ext import CASES, DOCKER_KINDS, FIXTURE, docker_ok, slm_record


def image():
    from siamlm import run as runner
    return str(runner.compose().ctx.bash.image)


@pytest.mark.parametrize("case", sorted(CASES))
def test_extended_requests_and_logs_are_unchanged(case):
    if CASES[case][1] in DOCKER_KINDS and not docker_ok(image()):
        pytest.skip(f"needs Docker and {image()}")
    gold = json.loads(FIXTURE.read_text(encoding="utf-8"))[case]
    got = slm_record([case])[case]
    assert len(gold["requests"]) == len(got["requests"])
    for k, (g, n) in enumerate(zip(gold["requests"], got["requests"])):
        assert len(g) == len(n), f"agent {k + 1}: {len(n)} calls, golden {len(g)}"
        for i, (gm, nm) in enumerate(zip(g, n)):
            assert gm == nm, f"agent {k + 1}, call {i + 1} differs"
    assert gold["tools"] == got["tools"]
    assert gold["logs"] == got["logs"]
    assert gold["files"] == got["files"]
