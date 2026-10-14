"""Command line interface: ``chatter index``, ``chatter search``, ``chatter ask``.

Exit codes: 0 on success (including "no results" and an abstaining answer),
1 on errors the user can fix (no index, model mismatch, model load failure).
"""

from __future__ import annotations

import enum
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from chatter.answer import (
    BACKENDS,
    DEFAULT_MODELS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PROMPT,
    PROMPTS,
    AnswerStatus,
    ContextTooSmallError,
    Generator,
    GeneratorConfig,
    load_generator,
    answer_question,
)
from chatter.answer_compare import (
    compare_answers,
    describe_grades,
    extract_grades,
    format_comparison,
    grades_path,
    load_criteria,
    load_grades,
    write_grades,
)
from chatter.answer_eval import (
    DEFAULT_ANSWER_TOP_K,
    format_answer_report,
    run_answer_eval,
    save_answer_results,
)
from chatter.evidence import Quote, describe_quote
from chatter.embed import DEFAULT_MODEL, Embedder, EmbedderConfig, SentenceTransformerEmbedder
from chatter.evaluate import (
    GRAPH_SWEEP_HOPS,
    SWEEP_DENSE_WEIGHT,
    SWEEP_RRF_K,
    EvalError,
    format_configs,
    format_graph_coverage,
    format_report,
    load_config_file,
    run_eval,
    save_results,
    sweep_configs,
    sweep_graph_configs,
)
from chatter.index import (
    IndexConfig,
    IndexMismatchError,
    build_index,
    default_index_dir,
    read_manifest,
)
from chatter.retrieve import DEFAULT_SEARCH, Hit, Retriever, SearchSettings, hit_location, hit_to_dict

Backend = enum.StrEnum("Backend", {name.upper(): name for name in BACKENDS})
_MODEL_HELP = "Hugging Face answer model. Default: " + ", ".join(
    f"{model} ({backend})" for backend, model in DEFAULT_MODELS.items()
)
_BACKEND_HELP = (
    "Answer generation backend. Default: mlx on Apple Silicon when mlx-lm is "
    "installed, otherwise transformers."
)


class SplitChoice(enum.StrEnum):
    ALL = "all"
    TUNE = "tune"
    HELDOUT = "heldout"


PromptChoice = enum.StrEnum("PromptChoice", {name.upper(): name for name in PROMPTS})
PromptOption = Annotated[
    PromptChoice,
    typer.Option(
        help="Answer prompt: plain, or evidence (quote supporting lines first; answers whose "
        "quotes can't be found in the cited chunks are withheld)."
    ),
]


class ModeChoice(enum.StrEnum):
    DENSE = "dense"
    FUSED = "fused"
    BM25 = "bm25"


ModeOption = Annotated[
    ModeChoice,
    typer.Option(help="Retriever: dense embeddings, fused BM25 + dense, or BM25 keywords only."),
]
GraphOption = Annotated[
    bool,
    typer.Option(
        "--graph/--no-graph",
        help="Add call-graph neighbours of the top results (dense and fused modes; never bm25).",
    ),
]
GraphHopsOption = Annotated[int, typer.Option("--graph-hops", min=1, help="Call-graph hops to expand.")]
GraphWeightOption = Annotated[
    float, typer.Option("--graph-weight", help="RRF weight of the call-graph list (> 0).")
]


EmbedderFactory = Callable[[str], Embedder]
GeneratorFactory = Callable[[GeneratorConfig], Generator]

INDEX_DIR_ENV = "CHATTER_INDEX_DIR"
IndexDirOption = Annotated[
    Path | None,
    typer.Option(
        "--index-dir",
        envvar=INDEX_DIR_ENV,
        show_envvar=True,
        help="Index directory. Default: <repo>/.chatter.",
    ),
]


def _default_embedder(model_name: str) -> Embedder:
    return SentenceTransformerEmbedder(EmbedderConfig(model_name=model_name))


def make_app(
    *,
    embedder_factory: EmbedderFactory = _default_embedder,
    generator_factory: GeneratorFactory = load_generator,
) -> typer.Typer:
    """Build the CLI. Factories are injectable so tests run offline."""
    app = typer.Typer(
        name="chatter",
        help="Index a Python repository and ask questions about it with file:line citations.",
        no_args_is_help=True,
        add_completion=False,
    )

    @app.command()
    def index(
        path: Annotated[Path, typer.Argument(help="Repository root to index.")],
        model: Annotated[
            str | None,
            typer.Option(help=f"Embedding model. Defaults to the index's model, else {DEFAULT_MODEL}."),
        ] = None,
        batch_size: Annotated[int, typer.Option(min=1, help="Texts per embedding call.")] = 256,
        rebuild: Annotated[
            bool, typer.Option(help="Drop stored vectors and re-embed (needed to change model).")
        ] = False,
        index_dir: IndexDirOption = None,
    ) -> None:
        """Build or update the index for PATH (stored in PATH/.chatter by default)."""
        if not path.is_dir():
            _fail(f"Not a directory: {path}")
        index_dir = index_dir or default_index_dir(path)
        manifest = read_manifest(index_dir)
        indexed_model = manifest.get("model") if manifest else None
        model_name = model or indexed_model or DEFAULT_MODEL
        if indexed_model and model_name != indexed_model and not rebuild:
            _fail(
                f"The index at {index_dir} was built with {indexed_model!r}.\n"
                f"Re-run with --rebuild to re-embed everything with {model_name!r}."
            )

        embedder = _load(lambda: embedder_factory(model_name), f"embedding model {model_name!r}")
        started = time.perf_counter()
        try:
            stats = build_index(
                path,
                embedder,
                index_dir=index_dir,
                config=IndexConfig(batch_size=batch_size),
                rebuild=rebuild,
            )
        except IndexMismatchError as exc:
            _fail(f"{exc}\nRe-run with --rebuild.")
        except OSError as exc:
            _fail(f"Cannot write the index at {index_dir}: {exc}")
        elapsed = time.perf_counter() - started
        typer.echo(
            f"Indexed {stats.files} files: {stats.chunks} chunks "
            f"({stats.embedded_parts} embedded, {stats.reused_chunks} unchanged, "
            f"{stats.deleted_chunks} deleted) in {elapsed:.1f}s"
        )
        typer.echo(f"Index: {index_dir}  model: {model_name}")

    @app.command()
    def search(
        query: Annotated[str, typer.Argument(help="Search query.")],
        k: Annotated[int, typer.Option("-k", "--top-k", min=1, help="Results to show.")] = 10,
        path: Annotated[Path, typer.Option(help="Repository root containing .chatter.")] = Path("."),
        as_json: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
        mode: ModeOption = ModeChoice(DEFAULT_SEARCH.mode),
        graph: GraphOption = DEFAULT_SEARCH.graph,
        graph_hops: GraphHopsOption = DEFAULT_SEARCH.graph_hops,
        graph_weight: GraphWeightOption = DEFAULT_SEARCH.graph_weight,
        index_dir: IndexDirOption = None,
    ) -> None:
        """Retrieve the most relevant chunks (no answer generation).

        Default: dense retrieval with 2-hop call-graph expansion, the config
        the held-out evaluation chose.
        """
        settings = _search_settings(mode, graph, graph_hops, graph_weight)
        hits = _search(_open_retriever(path, index_dir, embedder_factory), settings, query, k)
        if as_json:
            typer.echo(json.dumps([hit_to_dict(rank, hit) for rank, hit in enumerate(hits, 1)], indent=2))
            return
        if not hits:
            typer.echo("No results.")
            return
        for rank, hit in enumerate(hits, 1):
            typer.echo(
                f"{rank:>3}  {hit.score:.4f}  {'+'.join(hit.sources):<10}  "
                f"{hit_location(hit)}  {hit.chunk_id}"
            )

    @app.command()
    def ask(
        question: Annotated[str, typer.Argument(help="Question about the code.")],
        k: Annotated[int, typer.Option("-k", "--top-k", min=1, help="Chunks to retrieve.")] = 8,
        path: Annotated[Path, typer.Option(help="Repository root containing .chatter.")] = Path("."),
        model: Annotated[str | None, typer.Option(help=_MODEL_HELP)] = None,
        backend: Annotated[Backend | None, typer.Option(help=_BACKEND_HELP)] = None,
        max_context_tokens: Annotated[
            int, typer.Option(min=256, help="Token budget for the whole prompt.")
        ] = DEFAULT_MAX_CONTEXT_TOKENS,
        max_new_tokens: Annotated[
            int, typer.Option(min=16, help="Maximum answer length in tokens.")
        ] = DEFAULT_MAX_NEW_TOKENS,
        mode: ModeOption = ModeChoice(DEFAULT_SEARCH.mode),
        graph: GraphOption = DEFAULT_SEARCH.graph,
        graph_hops: GraphHopsOption = DEFAULT_SEARCH.graph_hops,
        graph_weight: GraphWeightOption = DEFAULT_SEARCH.graph_weight,
        prompt: PromptOption = PromptChoice(DEFAULT_PROMPT),
        index_dir: IndexDirOption = None,
    ) -> None:
        """Answer QUESTION from retrieved code, citing chunks.

        Retrieval defaults to dense with 2-hop call-graph expansion, as for search.
        """
        settings = _search_settings(mode, graph, graph_hops, graph_weight)
        hits = _search(_open_retriever(path, index_dir, embedder_factory), settings, question, k)
        if not hits:
            typer.echo("No indexed code matched the question; nothing to answer from.")
            return

        config = GeneratorConfig(model_name=model, backend=backend.value if backend else None)
        generator = _load(
            lambda: generator_factory(config),
            f"answer model {config.resolved_model!r} ({config.resolved_backend})",
            hint="`chatter search` still works without it.",
        )
        try:
            result = answer_question(
                question,
                hits,
                generator,
                max_context_tokens=max_context_tokens,
                max_new_tokens=max_new_tokens,
                prompt=prompt.value,
                on_text=lambda piece: typer.echo(piece, nl=False),
                on_evidence=_echo_evidence,
            )
        except ContextTooSmallError as exc:
            _fail(str(exc))
        typer.echo()

        if result.stopped_for_repetition:
            typer.echo("[generation stopped: the model started repeating itself]", err=True)
        if result.dropped:
            typer.echo(
                f"[{len(result.dropped)} lower-ranked chunk(s) did not fit --max-context-tokens]",
                err=True,
            )

        check = result.evidence
        if check is not None and check.forced_abstention:
            typer.echo(
                "NOT FOUND: none of the quoted lines appear in the cited chunks; "
                "the model's answer was withheld."
            )
            typer.echo("Searched:")
            for block in result.blocks:
                typer.echo(f"  [{block.tag}] {block.chunk_id}  {block.location}")
            return
        if check is not None and (warning := _partial_evidence_warning(check.quotes, check.unsupported_tags)):
            typer.echo(warning, err=True)
        if result.status is AnswerStatus.ABSTAINED:
            typer.echo("\nNOT FOUND: the retrieved code does not contain the answer.")
            typer.echo("Searched:")
            for block in result.blocks:
                typer.echo(f"  [{block.tag}] {block.chunk_id}  {block.location}")
            return
        if result.status is AnswerStatus.MIXED:
            typer.echo("\nNote: the model said the retrieved code does not fully answer this.")

        typer.echo("\nCitations:")
        if not result.citations:
            typer.echo("  (none - the answer cites no chunks)")
        for citation in result.citations:
            block = citation.block
            note = " (truncated)" if block.truncated else ""
            typer.echo(f"  [{citation.tag}] {block.chunk_id}  {block.location}{note}")
        if result.unknown_tags:
            typer.echo(
                f"Warning: the answer cites unknown tag(s): {', '.join(result.unknown_tags)}",
                err=True,
            )

    @app.command()
    def grade(
        review: Annotated[Path, typer.Argument(help="A filled-in answer review file (.md).")],
        out: Annotated[
            Path | None, typer.Option(help="Grades file. Default: eval/grades/<stem>.yaml next to eval/results/.")
        ] = None,
    ) -> None:
        """Collect the **Verdict:** lines of an answer review into a grades file."""
        if not review.is_file():
            _fail(f"No such file: {review}")
        grades, ungraded, problems = extract_grades(review.read_text(encoding="utf-8"))
        if problems:
            _fail("Cannot read the verdicts:\n  " + "\n  ".join(problems))
        results = review.with_suffix(".json")
        path = out or grades_path(results)
        write_grades(path, results, grades)
        typer.echo(f"{describe_grades(grades, ungraded)}\nWrote {path}")

    @app.command("compare-answers")
    def compare_answers_command(
        base: Annotated[Path, typer.Argument(help="Baseline answer results (.json).")],
        new: Annotated[Path, typer.Argument(help="New answer results (.json).")],
        criteria: Annotated[Path, typer.Option(help="Success criterion file (.yaml).")],
        base_grades: Annotated[
            Path | None, typer.Option(help="Default: eval/grades/<base stem>.yaml if it exists.")
        ] = None,
        new_grades: Annotated[
            Path | None, typer.Option(help="Default: eval/grades/<new stem>.yaml if it exists.")
        ] = None,
    ) -> None:
        """Score a new answer run against a baseline and a recorded success criterion."""
        try:
            spec = load_criteria(criteria)
            runs = [json.loads(path.read_text(encoding="utf-8")) for path in (base, new)]
            graded = [
                load_grades(path) if path is not None else None
                for path in (
                    base_grades or _existing(grades_path(base)),
                    new_grades or _existing(grades_path(new)),
                )
            ]
        except (EvalError, OSError, ValueError) as exc:
            _fail(str(exc))
        outcome = compare_answers(runs[0], runs[1], spec, base_grades=graded[0], new_grades=graded[1])
        typer.echo(format_comparison(outcome, str(base), str(new)))

    @app.command("mcp")
    def mcp(
        path: Annotated[
            Path | None,
            typer.Option(envvar="CHATTER_REPO", show_envvar=True, help="Repository root. Default: current directory."),
        ] = None,
        index_dir: IndexDirOption = None,
        enable_ask: Annotated[
            bool,
            typer.Option(
                envvar="CHATTER_MCP_ENABLE_ASK",
                show_envvar=True,
                help="Expose the ask tool (loads the local answer model on first use).",
            ),
        ] = False,
        backend: Annotated[
            Backend | None, typer.Option(envvar="CHATTER_BACKEND", show_envvar=True, help=_BACKEND_HELP)
        ] = None,
        answer_model: Annotated[
            str | None, typer.Option(envvar="CHATTER_ANSWER_MODEL", show_envvar=True, help=_MODEL_HELP)
        ] = None,
        answer_prompt: Annotated[
            PromptChoice,
            typer.Option(envvar="CHATTER_ANSWER_PROMPT", show_envvar=True, help="Answer prompt for ask."),
        ] = PromptChoice(DEFAULT_PROMPT),
    ) -> None:
        """Serve the index read-only to MCP clients (Claude Desktop, Claude Code) over stdio."""
        from chatter.mcp_server import ChatterService, check_index, config_from_env, run_stdio

        config = config_from_env(
            repo=path,
            index_dir=index_dir,
            enable_ask=enable_ask,
            backend=backend.value if backend else None,
            answer_model=answer_model,
            answer_prompt=answer_prompt.value,
            environ={},  # typer has already applied the environment variables
        )
        problem = check_index(config)
        if problem:
            _fail(problem)
        run_stdio(ChatterService(config, embedder_factory, generator_factory))

    @app.command("eval")
    def evaluate(
        questions: Annotated[Path, typer.Argument(help="Eval questions YAML.")],
        corpora: Annotated[
            Path | None, typer.Option(help="Corpus manifest. Default: corpora.yaml next to QUESTIONS.")
        ] = None,
        model: Annotated[str, typer.Option(help="Embedding model.")] = DEFAULT_MODEL,
        results_dir: Annotated[
            Path | None, typer.Option(help="Where to save results JSON. Default: results/ next to QUESTIONS.")
        ] = None,
        save: Annotated[bool, typer.Option(help="Save results JSON.")] = True,
        index_root: Annotated[
            Path | None,
            typer.Option(help="Store corpus indexes here (one subdirectory per corpus) "
                         "instead of inside each corpus."),
        ] = None,
        split: Annotated[
            SplitChoice, typer.Option(help="Only score questions in this split.")
        ] = SplitChoice.ALL,
        sweep_fusion: Annotated[
            bool,
            typer.Option(help="Also score a grid of RRF k x dense weight, with bm25-only, "
                         "dense-only and the current default as reference rows."),
        ] = False,
        rrf_k: Annotated[
            list[int] | None, typer.Option(help="RRF k values for --sweep-fusion (repeatable).")
        ] = None,
        dense_weight: Annotated[
            list[float] | None,
            typer.Option(help="Dense weights for --sweep-fusion (repeatable; BM25 weight is 1)."),
        ] = None,
        candidates: Annotated[
            Path | None,
            typer.Option(help="Also score the frozen retrieval configs in this YAML file."),
        ] = None,
        sweep_graph: Annotated[
            bool,
            typer.Option(help="Also score graph-expanded variants of the default and k10-w3 "
                         "(reference rows: dense-only, default, k10-w3)."),
        ] = False,
        graph_weight: Annotated[
            list[float] | None, typer.Option(help="Graph RRF weights for --sweep-graph (repeatable).")
        ] = None,
        graph_hops: Annotated[
            list[int] | None, typer.Option(help="Hops for --sweep-graph (repeatable).")
        ] = None,
        answers: Annotated[
            bool,
            typer.Option(help="Instead of retrieval metrics, answer every question with the local "
                         "model (default retrieval) and write a markdown review file."),
        ] = False,
        answer_model: Annotated[str | None, typer.Option(help=_MODEL_HELP)] = None,
        backend: Annotated[Backend | None, typer.Option(help=_BACKEND_HELP)] = None,
        answer_top_k: Annotated[
            int, typer.Option(min=1, help="Chunks retrieved per question for --answers.")
        ] = DEFAULT_ANSWER_TOP_K,
        max_context_tokens: Annotated[
            int, typer.Option(min=256, help="Prompt token budget for --answers.")
        ] = DEFAULT_MAX_CONTEXT_TOKENS,
        max_new_tokens: Annotated[
            int, typer.Option(min=16, help="Answer length limit for --answers.")
        ] = DEFAULT_MAX_NEW_TOKENS,
        prompt: PromptOption = PromptChoice(DEFAULT_PROMPT),
    ) -> None:
        """Score retrieval (bm25, dense, fused) on an eval set: hit@1, hit@5, MRR@50, recall@10."""
        if not questions.is_file():
            _fail(f"No such file: {questions}")
        if prompt.value != DEFAULT_PROMPT and not answers:
            _fail("--prompt applies to --answers runs only.")
        corpora_path = corpora or questions.parent / "corpora.yaml"
        if not corpora_path.is_file():
            _fail(f"No corpus manifest at {corpora_path}; pass --corpora.")
        if sum(map(bool, (sweep_fusion, candidates, sweep_graph))) > 1:
            _fail("Use only one of --sweep-fusion, --candidates, --sweep-graph.")
        if answers and (sweep_fusion or candidates or sweep_graph):
            _fail("--answers is a separate run; drop --sweep-fusion/--candidates.")
        if answers:
            answer_config = GeneratorConfig(
                model_name=answer_model, backend=backend.value if backend else None
            )
            try:
                answer_results = run_answer_eval(
                    questions,
                    corpora_path,
                    lambda name: _load(lambda: embedder_factory(name), f"embedding model {name!r}"),
                    lambda: generator_factory(answer_config),
                    model_name=model,
                    repo_root=_repo_root(questions.parent),
                    backend=answer_config.resolved_backend,
                    index_root=index_root,
                    split=None if split is SplitChoice.ALL else split.value,
                    top_k=answer_top_k,
                    max_context_tokens=max_context_tokens,
                    max_new_tokens=max_new_tokens,
                    prompt=prompt.value,
                    log=lambda line: typer.echo(line, err=True),
                )
            except EvalError as exc:
                _fail(f"Eval aborted: {exc}")
            typer.echo(format_answer_report(answer_results))
            if save:
                json_path, review_path = save_answer_results(
                    answer_results, results_dir or questions.parent / "results"
                )
                typer.echo(f"\nSaved {json_path}\nReview {review_path}")
            return
        decision_rules: list[dict[str, Any]] = []
        try:
            if sweep_fusion:
                configs = sweep_configs(rrf_k or list(SWEEP_RRF_K), dense_weight or list(SWEEP_DENSE_WEIGHT))
            elif sweep_graph:
                configs = sweep_graph_configs(graph_weight or None, graph_hops or GRAPH_SWEEP_HOPS)
            elif candidates:
                configs, decision_rules = load_config_file(candidates)
            else:
                configs = None
        except (EvalError, OSError) as exc:
            _fail(f"Eval aborted: {exc}")
        try:
            results = run_eval(
                questions,
                corpora_path,
                lambda name: _load(lambda: embedder_factory(name), f"embedding model {name!r}"),
                model_name=model,
                repo_root=_repo_root(questions.parent),
                index_root=index_root,
                split=None if split is SplitChoice.ALL else split.value,
                configs=configs,
                decision_rules=decision_rules,
                log=lambda line: typer.echo(line, err=True),
            )
        except EvalError as exc:
            _fail(f"Eval aborted: {exc}")
        typer.echo(format_report(results))
        if "configs" in results:
            typer.echo("\n" + format_configs(results["configs"]))
        typer.echo("\n" + format_graph_coverage(results["corpora"]))
        if save:
            path = save_results(results, results_dir or questions.parent / "results")
            typer.echo(f"\nSaved {path}")

    return app


def _repo_root(start: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=start, capture_output=True, text=True
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else start.resolve()


def _echo_evidence(quotes: Sequence[Quote]) -> None:
    if quotes:
        typer.echo("Evidence:")
        for quote in quotes:
            typer.echo(f"  {describe_quote(quote)}")
        typer.echo()


def _partial_evidence_warning(quotes: Sequence[Quote], unsupported: Sequence[str]) -> str | None:
    failed = sum(not q.verified for q in quotes)
    parts = [f"{failed} quote(s) could not be verified"] if failed else []
    if unsupported:
        parts.append(f"the answer cites {', '.join(unsupported)} without verified evidence")
    return f"[evidence partly verified: {'; '.join(parts)}]" if parts else None


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def _search_settings(mode: ModeChoice, graph: bool, graph_hops: int, graph_weight: float) -> SearchSettings:
    try:
        return SearchSettings(
            mode=mode.value, graph=graph, graph_hops=graph_hops, graph_weight=graph_weight  # type: ignore[arg-type]
        )
    except ValueError as exc:
        _fail(str(exc))


def _search(retriever: Retriever, settings: SearchSettings, query: str, k: int) -> list[Hit]:
    return retriever.search(query, k=k, mode=settings.mode, expansion=settings.expansion)


def _open_retriever(
    path: Path, index_dir: Path | None, embedder_factory: EmbedderFactory
) -> Retriever:
    default = default_index_dir(path)
    index_dir = index_dir or default
    manifest = read_manifest(index_dir)
    repo = (manifest or {}).get("repo", "<repo>") if index_dir != default else path
    command = f"chatter index {repo}" + ("" if index_dir == default else f" --index-dir {index_dir}")
    if manifest is None:
        _fail(f"No index found at {index_dir}.\nRun `{command}` first.")
    model_name = str(manifest.get("model", DEFAULT_MODEL))
    embedder = _load(lambda: embedder_factory(model_name), f"embedding model {model_name!r}")
    try:
        return Retriever.open(index_dir, embedder)
    except IndexMismatchError as exc:
        _fail(f"{exc}\nRe-run `{command} --rebuild`.")
    except FileNotFoundError as exc:
        _fail(f"Index at {index_dir} is incomplete ({exc}).\nRun `{command}`.")


def _load(factory: Callable[[], Any], what: str, *, hint: str = "") -> Any:
    try:
        return factory()
    except (OSError, ValueError, RuntimeError, ImportError, MemoryError) as exc:
        message = f"Could not load {what}: {exc}"
        _fail(f"{message}\n{hint}" if hint else message)


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


app = make_app()


def main() -> None:
    app()
