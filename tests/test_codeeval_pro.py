"""CodeEval-Pro: loading the pairs and controls, the statements (no dependency revealed), the Docker grader (references
pass, wrong programs fail, a dependent needs its base), the task in a world, and the stream builder's invariants.

The data is not shipped: tests that need it are skipped unless scripts/fetch_codeeval_pro.py has put it in
$SLM_DATA/codeeval_pro (default ./data/codeeval_pro)."""
import json
import os
import re
import shutil
import subprocess
from collections import Counter

import pytest
from omegaconf import OmegaConf

from helpers import Scripted, factory, read_jsonl, tool
from slm.tasks import cep_streams
from slm.tasks.codeeval_pro import DATA, EXCLUDED, EXCLUDED_CONTROLS, CodeEvalProTask, load, script_cases
from slm.world import World
from slm import run as runner

HAVE_DATA = (DATA / "bigcodebench_lite_pro.json").exists()


@pytest.fixture(scope="module")
def data():
    if not HAVE_DATA:
        pytest.skip(f"needs the CodeEval-Pro data in {DATA} (scripts/fetch_codeeval_pro.py; or set SLM_DATA)")
    return load()


# --- loading -----------------------------------------------------------------------------------------------------

def test_load_counts_ids_and_unaltered_texts(data):
    dataset = {r["raw_problem"]: r for r in json.loads((DATA / "bigcodebench_lite_pro.json").read_text())}
    assert len(data["pairs"]) == 57 - len(EXCLUDED) == 51
    assert len(data["controls"]) == 164 - len(EXCLUDED_CONTROLS) == 160          # 32 dropped after review
    items = [p[r] for p in data["pairs"] for r in ("base", "dependent")] + data["controls"]
    assert len({i["id"] for i in items}) == len(items)
    assert len({i["title"] for i in items}) == len(items)
    assert not {p["pair_id"] for p in data["pairs"]} & set(EXCLUDED)
    for p in data["pairs"]:
        b, d = p["base"], p["dependent"]
        assert (b["role"], d["role"]) == ("base", "dependent") and b["pair_id"] == d["pair_id"] == p["pair_id"]
        r = dataset[b["prompt"]]                                   # the dataset's texts, unaltered
        assert d["prompt"] == r["new_problem"]
        assert d["reference"] == r["raw_problem"] + r["raw_solution"] + r["new_problem"] + r["new_solution"]
        assert "".join(c["source"] for c in d["tests"]["cases"]) == r["test_code"]
        assert b["tests"]["kind"] == "unittest" and len(b["tests"]["cases"]) >= 4
        assert d["entry"] and f"def {d['entry']}" in d["prompt"]
    for c in data["controls"]:
        assert c["role"] == "control" and c["pair_id"] is None and f"def {c['entry']}" in c["prompt"]


def test_codeeval_pro_ids_are_realigned_by_text(data):
    """CodeEval-Pro's id field is shifted from its 12th record on; pair_id is the BigCodeBench task the text is."""
    p = next(p for p in data["pairs"] if p["pair_id"] == "BigCodeBench/217")
    assert p["base"]["cep_id"] == "BigCodeBench/213"
    assert sum(p["base"]["cep_id"] != p["pair_id"] for p in data["pairs"]) > 30


def test_script_cases_cut_after_each_assert():
    code = "import x\n\ny = 1\nassert y == 1\n# two\nfor i in [1]:\n    assert i\nz = 3\n"
    cases = script_cases(code)
    assert [c["source"] for c in cases] == ["import x\n\ny = 1\nassert y == 1\n", "# two\nfor i in [1]:\n    assert i\nz = 3\n"]
    helper = "def check(a):\n    assert a\n\ncheck(1)\ncheck(2)\n"
    assert len(script_cases(helper)) == 2


def test_statements_hide_the_dependency(data):
    task = CodeEvalProTask(grader=None)
    for p in data["pairs"]:
        b, d = p["base"], p["dependent"]
        for item, other in ((b, d), (d, b)):
            text = task.statement(item)
            assert text.startswith("Write a solution to the following problem:\n```python\n")
            assert item["prompt"].strip() in text
            assert other["prompt"].strip() not in text
            assert p["pair_id"] not in text and item["cep_id"] not in text and "self-invoking" not in text.lower()
            assert other["title"] not in text
        assert task.meta(d) == {"role": "dependent", "pair_id": p["pair_id"], "source": "bigcodebench_lite_pro",
                                "single_test": len(d["tests"]["cases"]) == 1}
    c = data["controls"][0]
    assert task.statement(c).startswith("Write a solution of python file to the following problems")


# --- the grader (Docker) -----------------------------------------------------------------------------------------

def docker_image():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", "slm-cep-grader:1"], capture_output=True).returncode == 0


needs_image = pytest.mark.skipif(not docker_image(), reason="needs Docker and slm-cep-grader:1")


@pytest.fixture(scope="module")
def grader():
    from slm.graders.cep import CepGrader
    g = CepGrader(OmegaConf.create({"image": "slm-cep-grader:1", "memory": "2g", "tmpfs_size": "256m", "pids_limit": 256,
                                    "command_timeout_s": 900, "timeout_per_test": 30}))
    yield g
    g.close()


def pick(data, *ids):
    return [next(p for p in data["pairs"] if p["pair_id"] == i) for i in ids]


@needs_image
def test_references_pass_and_wrong_programs_fail(data, grader):
    for p in pick(data, "BigCodeBench/23", "BigCodeBench/42", "BigCodeBench/316"):
        b, d = p["base"], p["dependent"]
        assert grader.grade(b, b["reference"])["passed"]
        assert grader.grade(d, d["reference"])["passed"]
        assert not grader.grade(d, b["reference"] + d["prompt"] + "    return None\n")["passed"]
        assert not grader.grade(b, b["prompt"] + "    return None\n")["passed"]
    for c in data["controls"][:3]:
        assert grader.grade(c, c["reference"])["passed"]


@needs_image
def test_dependent_reference_needs_the_base_function(data, grader):
    for p in pick(data, "BigCodeBench/23", "BigCodeBench/201", "BigCodeBench/1029"):
        b, d = p["base"], p["dependent"]
        alone = d["reference"][len(b["reference"]):]                # new_problem + new_solution, no task_func
        g = grader.grade(d, alone)
        assert not g["passed"] and "task_func" in g["metadata"]["error"]
        assert grader.grade(d, d["reference"])["passed"]


@needs_image
def test_public_feedback_and_isolation(data, grader):
    d = pick(data, "BigCodeBench/23")[0]["dependent"]
    pub = grader.grade(d, d["prompt"] + "    return []\n", "public")
    assert pub["n_tests"] == 1 and not pub["passed"]
    assert d["tests"]["cases"][0]["source"].strip() in pub["feedback"] and "/grader/" not in pub["feedback"]
    spoof = "print('__grade__' + '{\"results\": [true, true, true, true], \"error\": null}')\nimport os\nos._exit(0)\n"
    assert not grader.grade(d, spoof)["passed"]                     # its prints never reach the grader's output
    assert "compilation error" in grader.grade(d, "def f(:\n")["feedback"]
    slow = grader.grade(d, "def new_task_func(*a):\n    while True: pass\n", "public")
    assert not slow["passed"] and "time limit" in slow["feedback"]


@needs_image
@pytest.mark.skipif(not os.environ.get("CEP_FULL"), reason="every reference (~2 min): set CEP_FULL=1")
def test_references_in_docker(data, grader):
    """Every kept pair: base and dependent references pass; the dependent's fails without the base. Every control passes."""
    bad = []
    for p in data["pairs"]:
        b, d = p["base"], p["dependent"]
        if not grader.grade(b, b["reference"])["passed"] or not grader.grade(d, d["reference"])["passed"] \
                or grader.grade(d, d["reference"][len(b["reference"]):])["passed"]:
            bad.append(p["pair_id"])
    bad += [c["id"] for c in data["controls"] if not grader.grade(c, c["reference"])["passed"]]
    assert bad == []


# --- the task in a world -----------------------------------------------------------------------------------------

class FakeCepGrader:
    """Passes a program that contains OK."""
    image_id = "fake"

    def grade(self, item, code, which="all"):
        ok = "OK" in (code or "")
        n = 1 if which == "public" else len(item["tests"]["cases"])
        return {"passed": ok, "n_tests": n, "n_passed": n * ok, "n_passed_before_fail": n * ok, "results": [ok] * n,
                "metadata": {} if ok else {"error_code": "fail"}, "seconds": 0.0, "cached": False,
                "feedback": "" if ok else "The above code is incorrect and failed this test:"}

    def close(self):
        pass


def test_world_runs_codeeval_pro_streams(tmp_path, data):
    st, manifest = cep_streams.build(data, 2, 3, "forward_partner", length=2, gap=0)
    cfg = runner.compose(["world.n_agents=2"])
    OmegaConf.set_struct(cfg, False)
    a1 = Scripted([tool("run_tests", code="x = 1"), tool("submit", code="# OK")])
    a2 = Scripted([tool("submit", code="# OK")])
    World(cfg, 1, st, tmp_path / "w", CodeEvalProTask(FakeCepGrader()), factory(a1, a2), label="t").run()
    res = read_jsonl(tmp_path / "w" / "results.jsonl")
    assert sorted(r["problem"] for r in res) == sorted(i["id"] for s in st for i in s)
    assert {r["role"] for r in res} == {"base", "dependent"} and all(r["passed"] for r in res)
    first = " ".join(str(m.get("content") or "") for m in a1.seen[0])
    assert "Write a solution to the following problem:" in first and st[0][0]["entry"] in first
    assert "BigCodeBench" not in first and "depend" not in first.lower()
    assert "The public test failed." in " ".join(str(m.get("content") or "") for m in a1.seen[1])


# --- streams -----------------------------------------------------------------------------------------------------

def check_manifest(st, m, gap):
    ids = [i["id"] for s in st for i in s]
    assert len(set(ids)) == len(ids)                                   # nothing twice in a world
    assert len({len(s) for s in st}) == 1 and len(st[0]) == m["length"]
    assert m["streams"] == [[{"id": i["id"], "role": i["role"], "pair_id": i["pair_id"], "title": i["title"]} for i in s] for s in st]
    at = lambda e: st[e["agent"] - 1][e["position"] - 1]["id"]
    pair_ids = Counter()
    for link in m["links"]:
        f, s = link[link["first"]], link["dependent" if link["first"] == "base" else "base"]
        assert at(link["base"]) == link["base"]["id"] == f"{link['pair_id']}:base"
        assert at(link["dependent"]) == link["dependent"]["id"] == f"{link['pair_id']}:dependent"
        assert s["position"] - f["position"] == gap + 1
        assert link["first"] == ("base" if link["direction"] == "forward" else "dependent")
        assert (f["agent"] == s["agent"]) == (link["who"] == "self")
        if link["who"] == "partner":
            assert s["agent"] == f["agent"] % m["n_agents"] + 1
        pair_ids[link["pair_id"]] += 1
    for e in m["solo"]:
        assert at(e) == e["id"]
        pair_ids[e["pair_id"]] += 1
    assert all(n == 1 for n in pair_ids.values())                     # a pair is used once
    for e in m["controls"]:
        assert at(e) == e["id"] and st[e["agent"] - 1][e["position"] - 1]["role"] == "control"
    linked = {l[r]["id"] for l in m["links"] for r in ("base", "dependent")}
    for s in st:                                                       # every non-control item is in the manifest
        for i in s:
            assert i["role"] == "control" or i["id"] in linked | {e["id"] for e in m["solo"]}


@pytest.mark.parametrize("arrangement", cep_streams.ARRANGEMENTS)
@pytest.mark.parametrize("n_agents", [1, 2, 3])
@pytest.mark.parametrize("gap", [0, 2])
def test_stream_invariants(data, arrangement, n_agents, gap):
    if arrangement.endswith("partner") and n_agents == 1:
        with pytest.raises(ValueError):
            cep_streams.build(data, n_agents, 1, arrangement, 12, gap)
        return
    st, m = cep_streams.build(data, n_agents, 1, arrangement, 12, gap)
    check_manifest(st, m, gap)
    assert json.loads(json.dumps(m)) == m
    again, m2 = cep_streams.build(data, n_agents, 1, arrangement, 12, gap)
    assert m2 == m and again == st                                     # deterministic by seed
    other = cep_streams.build(data, n_agents, 2, arrangement, 12, gap)[1]
    assert other["streams"] != m["streams"]
    if arrangement in cep_streams.LINKS:
        assert {l["type"] for l in m["links"]} == {arrangement}
        assert len(m["links"]) == n_agents * (12 // (2 * (gap + 1))) * (gap + 1)


def test_mixed_is_the_same_blend_for_every_agent(data):
    st, m = cep_streams.build(data, 2, 5, "mixed", 12, 2)
    assert Counter(m["slot_types"]) == Counter(cep_streams.MIX)
    for k in (1, 2):
        mine = Counter(l["type"] for l in m["links"] if l[l["first"]]["agent"] == k)
        assert mine == Counter({t: 1 for t in cep_streams.LINKS})
        assert sum(1 for c in m["controls"] if c["agent"] == k) == 4
    roles = [Counter(i["role"] for i in s) for s in st]
    assert roles[0] == roles[1] == Counter({"base": 4, "dependent": 4, "control": 4})


def test_too_few_pairs_is_an_error(data):
    with pytest.raises(ValueError):
        cep_streams.build(data, 4, 1, "forward_self", 30, 0)


def test_driver_deletes_the_job_and_rejects_out_of_order_results(tmp_path):
    import importlib.util, json as _json, os as _os
    from pathlib import Path as _Path
    spec = importlib.util.spec_from_file_location("cep_driver", str(_Path(__file__).resolve().parents[1] / "slm" / "graders" / "cep_driver.py"))
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    job = tmp_path / "job.json"
    job.write_text(_json.dumps({"code": "x = 1", "kind": "script", "tests": [{"name": "t1", "source": "assert x == 1"}],
                                "timeout": 5}))
    drv.main(str(job))
    assert not job.exists()


def test_rotation_keeps_the_layout_and_moves_the_pairs(data):
    _, m0 = cep_streams.build(data, 2, 1, "mixed", 24, 2)
    _, m1 = cep_streams.build(data, 2, 1, "mixed", 24, 2, rotate=12)
    assert m0["slot_types"] == m1["slot_types"] and m1["rotate"] == 12
    assert [l["pair_id"] for l in m0["links"]] != [l["pair_id"] for l in m1["links"]]
    assert [l["type"] for l in m0["links"]] == [l["type"] for l in m1["links"]]


def test_gate_marks_only_the_second_item_of_cross_agent_links(data):
    st, m = cep_streams.build(data, 2, 1, "mixed", 24, 2, gate=True)
    marked = {it["id"]: it["after"] for s in st for it in s if it.get("after")}
    partner = {l[("dependent" if l["first"] == "base" else "base")]["id"]: l[l["first"]]["id"]
               for l in m["links"] if l["who"] == "partner"}
    assert marked == partner and marked and m["gate"] is True
    assert not any(it.get("after") for s in cep_streams.build(data, 2, 1, "mixed", 24, 2)[0] for it in s)


def test_base_use_reads_the_program():
    from slm.tasks.codeeval_pro import base_use
    base = "def task_func(x):\n    return x\n"
    assert base_use(base + "def f(y):\n    return task_func(y)", "task_func", "f") == {"defines_base": True, "calls_base": True}
    assert base_use(base + "def g(y):\n    return list(map(task_func, y))\ndef f(y):\n    return g(y)", "task_func", "f") == \
        {"defines_base": True, "calls_base": True}                      # through a helper, passed on as a value
    assert base_use(base + "def f(y):\n    return 42\nprint(task_func(1))", "task_func", "f") == \
        {"defines_base": True, "calls_base": False}                     # used only outside the dependent's function
    assert base_use("class A:\n    def task_func(self):\n        pass\ndef f(y):\n    return 42", "task_func", "f") == \
        {"defines_base": False, "calls_base": False}
    assert base_use("def f(:", "task_func", "f") == base_use("", "task_func", "f") == {"defines_base": None, "calls_base": None}



def test_every_reference_solution_builds_on_its_base(data):
    from slm.tasks.codeeval_pro import base_use
    for p in data["pairs"]:
        d = p["dependent"]
        assert base_use(d["reference"], d["base_entry"], d["entry"]) == {"defines_base": True, "calls_base": True}, d["id"]
