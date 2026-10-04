from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from chatter.cli import make_app
from chatter.evaluate import (
    CORPUS_MARKER,
    DEPTH,
    CorpusSpec,
    EvalError,
    Question,
    aggregate,
    format_report,
    is_test_chunk,
    load_questions,
    prepare_corpus,
    run_eval,
    save_results,
    score_ranking,
    summarize,
    validate_relevant,
)
from conftest import HashEmbedder

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def q(qid: str = "q1", qtype: str = "behavior", relevant: tuple[str, ...] = ("a.py::f",)) -> Question:
    return Question(qid, "c", qtype, "question?", relevant, qtype == "negative")


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def make_git_corpus(tmp_path: Path, files: dict[str, str], tag: str = "v1") -> Path:
    repo = tmp_path / "src_repo"
    repo.mkdir(exist_ok=True)
    git(repo, "init", "-q")
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip("\n"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "corpus")
    git(repo, "tag", tag)
    return repo


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_score_ranking_metrics() -> None:
    ranked = ["x", "tests/t.py::t", "a.py::f", "y", "conftest.py::c", "a.py::g"] + ["z"] * 10
    s = score_ranking(ranked, ["a.py::f", "a.py::g", "a.py::h"])
    assert (s.first_rank, s.hit1, s.hit5) == (3, False, True)
    assert s.rr == pytest.approx(1 / 3)
    assert s.recall10 == pytest.approx(2 / 3)
    assert s.test_slots5 == 2


def test_score_ranking_top1_miss_and_depth_cutoff() -> None:
    assert score_ranking(["a"], ["a"]).hit1
    filler = [f"n{i}" for i in range(DEPTH)]
    s = score_ranking([*filler, "a"], ["a"])
    assert (s.first_rank, s.rr, s.hit5, s.recall10) == (None, 0.0, False, 0.0)
    assert len(s.top) == DEPTH
    assert score_ranking([], ["a"]).first_rank is None


@pytest.mark.parametrize(
    ("chunk_id", "expected"),
    [
        ("tests/test_x.py::f", True),
        ("pkg/tests/helpers.py::f", True),
        ("test_api.py::f", True),
        ("api_test.py::f", True),
        ("tests/conftest.py::FakeGenerator", True),
        ("src/chatter/answer.py::RepetitionGuard", False),
        ("email/test.py::f", False),  # a module named test.py is not a test dir
        ("src/testing_utils.py::f", False),
    ],
)
def test_is_test_chunk(chunk_id: str, expected: bool) -> None:
    assert is_test_chunk(chunk_id) is expected


def test_aggregate_excludes_negatives_and_groups_by_type() -> None:
    hit = score_ranking(["a.py::f"], ["a.py::f"])
    miss = score_ranking(["x"], ["a.py::f"])
    rows = [
        (q("q1", "behavior"), hit),
        (q("q2", "behavior"), miss),
        (q("q3", "flow"), hit),
        (q("q4", "negative", ()), score_ranking(["tests/t.py::t"], [])),
    ]
    overall = aggregate(rows)
    assert overall["n"] == 3
    assert overall["hit@1"] == pytest.approx(2 / 3) and overall["mrr@50"] == pytest.approx(2 / 3)
    assert overall["test_slots@5"] == 0 and overall["slots@5"] == 3
    summary = summarize(rows)
    assert set(summary["by_type"]) == {"behavior", "flow"}
    assert summary["by_type"]["behavior"]["hit@1"] == 0.5
    assert aggregate([(q("q4", "negative", ()), miss)]) == {"n": 0}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_load_questions_reports_every_problem(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {"id": "a", "corpus": "c", "type": "behavior", "question": "?", "relevant": ["x"]},
                {"id": "a", "corpus": "c", "type": "behavior", "question": "?", "relevant": ["x"]},
                {"id": "b", "corpus": "c", "type": "vibes", "question": "?"},
                {"id": "c", "corpus": "c", "type": "negative", "question": "?", "relevant": ["x"]},
                {"id": "d", "corpus": "c", "type": "flow", "question": "?", "relevant": []},
                {"id": "e", "type": "flow"},
            ]
        )
    )
    with pytest.raises(EvalError) as err:
        load_questions(path)
    message = str(err.value)
    for expected in ("a: duplicate id", "b: unknown type", "c: negative", "d: no relevant", "missing corpus, question"):
        assert expected in message


def test_validate_relevant_fails_loudly_on_stale_ids() -> None:
    questions = [q("q1", relevant=("a.py::f", "a.py::gone")), q("q2", relevant=("b.py::old",))]
    with pytest.raises(EvalError) as err:
        validate_relevant(questions, {"c": {"a.py::f"}})
    assert "q1: a.py::gone" in str(err.value) and "q2: b.py::old" in str(err.value)
    validate_relevant([q("q1")], {"c": {"a.py::f"}})


def test_repository_eval_set_is_well_formed() -> None:
    questions = load_questions(EVAL_DIR / "questions.yaml")
    assert len(questions) == 22 and len({x.id for x in questions}) == 22
    negatives = [x for x in questions if x.type == "negative"]
    assert [x.id for x in negatives] == ["q15", "q22"]
    assert all(x.expect_abstain for x in negatives)
    assert all(x.relevant for x in questions if x.scored)
    corpora = yaml.safe_load((EVAL_DIR / "corpora.yaml").read_text())
    assert {x.corpus for x in questions} == set(corpora)
    assert corpora["chatter"]["source"] == {"kind": "git_archive", "ref": "eval-corpus-v1"}


# ---------------------------------------------------------------------------
# Corpus materialisation
# ---------------------------------------------------------------------------


def test_git_archive_corpus_snapshot_and_reuse(tmp_path: Path) -> None:
    repo = make_git_corpus(tmp_path, {"pkg/a.py": "def f():\n    pass\n"})
    spec = CorpusSpec("c", tmp_path / "corpus", {"kind": "git_archive", "ref": "v1"})
    info = prepare_corpus(spec, repo_root=repo)
    assert info == {"kind": "git_archive", "ref": "v1", "commit": git(repo, "rev-parse", "v1")}
    assert (spec.path / "pkg" / "a.py").read_text() == "def f():\n    pass\n"
    assert json.loads((spec.path / CORPUS_MARKER).read_text()) == info

    (spec.path / "sentinel.txt").write_text("kept")  # unchanged ref: no re-export
    (repo / "pkg" / "a.py").write_text("def changed():\n    pass\n")  # live tree edits ignored
    assert prepare_corpus(spec, repo_root=repo) == info
    assert (spec.path / "sentinel.txt").exists()
    assert "def f" in (spec.path / "pkg" / "a.py").read_text()

    git(repo, "commit", "-q", "-am", "next")
    git(repo, "tag", "v2")
    moved = prepare_corpus(
        CorpusSpec("c", spec.path, {"kind": "git_archive", "ref": "v2"}), repo_root=repo
    )
    assert moved["ref"] == "v2" and not (spec.path / "sentinel.txt").exists()
    assert "def changed" in (spec.path / "pkg" / "a.py").read_text()


def test_git_archive_missing_ref(tmp_path: Path) -> None:
    repo = make_git_corpus(tmp_path, {"a.py": "x = 1\n"})
    spec = CorpusSpec("c", tmp_path / "corpus", {"kind": "git_archive", "ref": "nope"})
    with pytest.raises(EvalError, match="'nope' not found"):
        prepare_corpus(spec, repo_root=repo)


def test_stdlib_corpus_copies_packages_without_bytecode(tmp_path: Path) -> None:
    stdlib = tmp_path / "lib"
    for pkg in ("json", "email", "other"):
        (stdlib / pkg / "__pycache__").mkdir(parents=True)
        (stdlib / pkg / "__init__.py").write_text(f"NAME = {pkg!r}\n")
        (stdlib / pkg / "__pycache__" / "x.cpython-313.pyc").write_bytes(b"\0")
    spec = CorpusSpec(
        "s", tmp_path / "corpus",
        {"kind": "python_stdlib", "python_version": "3.13", "packages": ["json", "email"]},
    )
    info = prepare_corpus(spec, repo_root=tmp_path, stdlib_dir=stdlib)
    assert sorted(p.name for p in spec.path.iterdir() if not p.name.startswith(".")) == ["email", "json"]
    assert not list(spec.path.rglob("*.pyc"))
    assert info["packages"] == ["json", "email"] and info["stdlib_dir"] == str(stdlib)


def test_stdlib_corpus_requires_matching_python(tmp_path: Path) -> None:
    spec = CorpusSpec(
        "s", tmp_path / "corpus", {"kind": "python_stdlib", "python_version": "2.7", "packages": ["json"]}
    )
    with pytest.raises(EvalError, match="needs Python 2.7"):
        prepare_corpus(spec, repo_root=tmp_path)


def test_unknown_corpus_kind(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="unknown source kind"):
        prepare_corpus(CorpusSpec("c", tmp_path / "x", {"kind": "ftp"}), repo_root=tmp_path)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


CORPUS_FILES = {
    "lib/cache.py": '''
        def evict_least_recently_used(cache, capacity):
            """Drop the oldest entries until the cache fits."""
            while len(cache) > capacity:
                cache.popitem(last=False)

        def warm_cache(cache, keys):
            for key in keys:
                cache[key] = None
    ''',
    "lib/net.py": '''
        def parse_http_response(raw):
            head, _, body = raw.partition("\\r\\n\\r\\n")
            return head, body
    ''',
    "tests/test_cache.py": '''
        def test_evict_least_recently_used():
            assert True
    ''',
}

QUESTIONS = [
    {"id": "e1", "corpus": "mini", "type": "exact_name", "question": "evict_least_recently_used",
     "relevant": ["lib/cache.py::evict_least_recently_used"]},
    {"id": "e2", "corpus": "mini", "type": "flow", "question": "parse http response head body",
     "relevant": ["lib/net.py::parse_http_response", "lib/cache.py::warm_cache"]},
    {"id": "e3", "corpus": "mini", "type": "negative", "question": "oauth tokens", "relevant": [],
     "expect_abstain": True},
]


def write_eval(tmp_path: Path, questions: list[dict[str, object]] = QUESTIONS) -> tuple[Path, Path, Path]:
    repo = make_git_corpus(tmp_path, CORPUS_FILES)
    eval_dir = repo / "eval"  # as in the real layout: eval/ lives inside the repo
    eval_dir.mkdir()
    (eval_dir / "questions.yaml").write_text(yaml.safe_dump(questions, sort_keys=False))
    (eval_dir / "corpora.yaml").write_text(
        yaml.safe_dump({"mini": {"path": ".corpora/mini", "source": {"kind": "git_archive", "ref": "v1"}}})
    )
    return repo, eval_dir / "questions.yaml", eval_dir / "corpora.yaml"


def test_run_eval_end_to_end(tmp_path: Path) -> None:
    repo, questions, corpora = write_eval(tmp_path)
    results = run_eval(
        questions, corpora, lambda name: HashEmbedder(name=name), model_name="fake-embed", repo_root=repo
    )
    assert results["embedding_model"] == "fake-embed"
    assert results["git"] == {"commit": git(repo, "rev-parse", "HEAD"), "dirty": False}
    assert results["corpora"]["mini"]["ref"] == "v1" and results["corpora"]["mini"]["chunks"] == 4
    assert set(results["summary"]) == {"bm25", "dense", "fused"}
    overall = results["summary"]["fused"]["overall"]
    assert overall["n"] == 2  # the negative question is excluded
    by_id = {x["id"]: x for x in results["questions"]}
    e1 = by_id["e1"]["modes"]
    assert e1["bm25"]["first_rank"] == 1 and e1["fused"]["hit1"]
    assert e1["bm25"]["test_slots5"] == 1  # the test function competes for a top-5 slot
    assert by_id["e3"]["scored"] is False and by_id["e3"]["modes"]["fused"]["top10"]
    assert by_id["e2"]["modes"]["bm25"]["recall10"] == 0.5  # warm_cache has no query terms

    report = format_report(results)
    assert "fused  overall" in report and "e3   negative" in report and "n/a" in report

    again = run_eval(questions, corpora, lambda n: HashEmbedder(name=n), model_name="fake-embed", repo_root=repo)
    assert again["summary"] == results["summary"]  # deterministic


def test_run_eval_fails_on_stale_relevant_id(tmp_path: Path) -> None:
    stale = [{**QUESTIONS[0], "relevant": ["lib/cache.py::renamed_function"]}]
    repo, questions, corpora = write_eval(tmp_path, stale)
    with pytest.raises(EvalError, match="e1: lib/cache.py::renamed_function"):
        run_eval(questions, corpora, lambda n: HashEmbedder(name=n), model_name="m", repo_root=repo)


def test_run_eval_rebuilds_index_for_new_model(tmp_path: Path) -> None:
    repo, questions, corpora = write_eval(tmp_path)
    run_eval(questions, corpora, lambda n: HashEmbedder(name=n), model_name="m1", repo_root=repo)
    results = run_eval(questions, corpora, lambda n: HashEmbedder(name=n, dim=32), model_name="m2", repo_root=repo)
    assert results["embedding_model"] == "m2"


def test_save_results_names_file_by_commit(tmp_path: Path) -> None:
    results = {"created": "2026-10-04T20:31:05+00:00", "git": {"commit": "abcdef1234", "dirty": True}}
    path = save_results(results, tmp_path / "results")
    assert path.name == "20261004T203105-abcdef1-dirty.json"
    assert json.loads(path.read_text()) == results


def test_cli_eval_prints_table_and_saves(tmp_path: Path) -> None:
    repo, questions, _ = write_eval(tmp_path)
    app = make_app(embedder_factory=lambda name: HashEmbedder(name=name))
    result = CliRunner().invoke(app, ["eval", str(questions)])
    assert result.exit_code == 0, result.output
    assert "hit@1" in result.output and "MRR@50" in result.output and "R@10" in result.output
    saved = list((questions.parent / "results").glob("*.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text())
    assert data["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert data["corpora"]["mini"]["commit"] == git(repo, "rev-parse", "v1")


def test_cli_eval_errors(tmp_path: Path) -> None:
    app = make_app(embedder_factory=lambda name: HashEmbedder(name=name))
    runner = CliRunner()
    missing = runner.invoke(app, ["eval", str(tmp_path / "nope.yaml")])
    assert missing.exit_code == 1 and "No such file" in missing.output

    stale = [{**QUESTIONS[0], "relevant": ["lib/cache.py::gone"]}]
    _, questions, _ = write_eval(tmp_path, stale)
    aborted = runner.invoke(app, ["eval", str(questions), "--no-save"])
    assert aborted.exit_code == 1 and "Eval aborted" in aborted.output and "gone" in aborted.output
    assert not (questions.parent / "results").exists()


def test_run_eval_index_root_keeps_indexes_out_of_corpus(tmp_path: Path) -> None:
    repo, questions, corpora = write_eval(tmp_path)
    root = tmp_path / "indexes"
    results = run_eval(
        questions, corpora, lambda n: HashEmbedder(name=n), model_name="m", repo_root=repo, index_root=root
    )
    assert results["summary"]["fused"]["overall"]["n"] == 2
    assert (root / "mini" / "manifest.json").exists()
    assert not (questions.parent / ".corpora" / "mini" / ".chatter").exists()


def test_cli_eval_reports_unwritable_index_storage(tmp_path: Path) -> None:
    import os

    _, questions, _ = write_eval(tmp_path)
    locked = tmp_path / "locked"
    (locked / "mini").mkdir(parents=True)
    (locked / "mini").chmod(0o500)
    try:
        if os.access(locked / "mini", os.W_OK):
            pytest.skip("running with privileges that ignore directory permissions")
        app = make_app(embedder_factory=lambda name: HashEmbedder(name=name))
        result = CliRunner().invoke(app, ["eval", str(questions), "--index-root", str(locked), "--no-save"])
        assert result.exit_code == 1
        assert "SQLite cannot write" in result.output and "--index-root" in result.output
    finally:
        (locked / "mini").chmod(0o700)
