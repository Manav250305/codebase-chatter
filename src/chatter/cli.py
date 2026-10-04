"""Command line interface: ``chatter index``, ``chatter search``, ``chatter ask``.

Exit codes: 0 on success (including "no results" and an abstaining answer),
1 on errors the user can fix (no index, model mismatch, model load failure).
"""

from __future__ import annotations

import enum
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from chatter.answer import (
    DEFAULT_ANSWER_MODEL,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    AnswerStatus,
    ContextTooSmallError,
    Generator,
    GeneratorConfig,
    HFGenerator,
    answer_question,
    format_line_ranges,
)
from chatter.embed import DEFAULT_MODEL, Embedder, EmbedderConfig, SentenceTransformerEmbedder
from chatter.evaluate import (
    SWEEP_DENSE_WEIGHT,
    SWEEP_RRF_K,
    EvalError,
    format_configs,
    format_report,
    load_configs,
    run_eval,
    save_results,
    sweep_configs,
)
from chatter.index import (
    IndexConfig,
    IndexMismatchError,
    IndexStorageError,
    build_index,
    default_index_dir,
    read_manifest,
)
from chatter.retrieve import Hit, Retriever

class SplitChoice(enum.StrEnum):
    ALL = "all"
    TUNE = "tune"
    HELDOUT = "heldout"


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
    generator_factory: GeneratorFactory = HFGenerator,
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
        except IndexStorageError as exc:
            _fail(str(exc))
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
        index_dir: IndexDirOption = None,
    ) -> None:
        """Retrieve the most relevant chunks (no answer generation)."""
        hits = _open_retriever(path, index_dir, embedder_factory).search(query, k=k)
        if as_json:
            typer.echo(json.dumps([_hit_json(rank, hit) for rank, hit in enumerate(hits, 1)], indent=2))
            return
        if not hits:
            typer.echo("No results.")
            return
        for rank, hit in enumerate(hits, 1):
            typer.echo(
                f"{rank:>3}  {hit.score:.4f}  {'+'.join(hit.sources):<10}  "
                f"{_hit_location(hit)}  {hit.chunk_id}"
            )

    @app.command()
    def ask(
        question: Annotated[str, typer.Argument(help="Question about the code.")],
        k: Annotated[int, typer.Option("-k", "--top-k", min=1, help="Chunks to retrieve.")] = 8,
        path: Annotated[Path, typer.Option(help="Repository root containing .chatter.")] = Path("."),
        model: Annotated[str, typer.Option(help="Hugging Face answer model.")] = DEFAULT_ANSWER_MODEL,
        max_context_tokens: Annotated[
            int, typer.Option(min=256, help="Token budget for the whole prompt.")
        ] = DEFAULT_MAX_CONTEXT_TOKENS,
        max_new_tokens: Annotated[
            int, typer.Option(min=16, help="Maximum answer length in tokens.")
        ] = DEFAULT_MAX_NEW_TOKENS,
        index_dir: IndexDirOption = None,
    ) -> None:
        """Answer QUESTION from retrieved code, citing chunks."""
        hits = _open_retriever(path, index_dir, embedder_factory).search(question, k=k)
        if not hits:
            typer.echo("No indexed code matched the question; nothing to answer from.")
            return

        generator = _load(
            lambda: generator_factory(GeneratorConfig(model_name=model)),
            f"answer model {model!r}",
            hint="`chatter search` still works without it.",
        )
        try:
            result = answer_question(
                question,
                hits,
                generator,
                max_context_tokens=max_context_tokens,
                max_new_tokens=max_new_tokens,
                on_text=lambda piece: typer.echo(piece, nl=False),
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
                         "instead of inside each corpus, e.g. when the repo is on exFAT."),
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
    ) -> None:
        """Score retrieval (bm25, dense, fused) on an eval set: hit@1, hit@5, MRR@50, recall@10."""
        if not questions.is_file():
            _fail(f"No such file: {questions}")
        corpora_path = corpora or questions.parent / "corpora.yaml"
        if not corpora_path.is_file():
            _fail(f"No corpus manifest at {corpora_path}; pass --corpora.")
        if sweep_fusion and candidates:
            _fail("Use either --sweep-fusion or --candidates, not both.")
        try:
            configs = (
                sweep_configs(rrf_k or list(SWEEP_RRF_K), dense_weight or list(SWEEP_DENSE_WEIGHT))
                if sweep_fusion
                else load_configs(candidates)
                if candidates
                else None
            )
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
                log=lambda line: typer.echo(line, err=True),
            )
        except EvalError as exc:
            _fail(f"Eval aborted: {exc}")
        typer.echo(format_report(results))
        if "configs" in results:
            typer.echo("\n" + format_configs(results["configs"]))
        if save:
            path = save_results(results, results_dir or questions.parent / "results")
            typer.echo(f"\nSaved {path}")

    return app


def _repo_root(start: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=start, capture_output=True, text=True
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else start.resolve()


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


def _hit_location(hit: Hit) -> str:
    chunk = hit.chunk
    if chunk.spans and hit.lines == (chunk.start_line, chunk.end_line):
        return f"{chunk.path}:{format_line_ranges(chunk.line_numbers())}"
    return f"{chunk.path}:{hit.lines[0]}-{hit.lines[1]}"


def _hit_json(rank: int, hit: Hit) -> dict[str, Any]:
    chunk = hit.chunk
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "path": chunk.path,
        "qualname": chunk.qualname,
        "kind": chunk.kind,
        "lines": list(hit.lines),
        "spans": [list(span) for span in (chunk.spans or ((chunk.start_line, chunk.end_line),))],
        "score": hit.score,
        "sources": list(hit.sources),
        "ranks": hit.ranks,
        "raw_scores": hit.raw_scores,
    }


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


app = make_app()


def main() -> None:
    app()
