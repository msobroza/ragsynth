"""Typer CLI: ``ragsynth run|validate|report --config <yaml>`` (SPEC §5).

The only module allowed to talk to the terminal (rich console); everything
else logs. Importing this module imports all step/adapter/dataset/arm
concretes so the registries are fully populated for config resolution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from ragsynth.domain import EvalReport
from ragsynth.pipeline.registry import RegistryError
from ragsynth.pipeline.serialization import (
    build_pipeline,
    build_resources,
    load_config,
    make_initial_state,
    validate_config,
)
from ragsynth.steps.validator import render_figures, render_markdown

app = typer.Typer(
    help="Synthetic query generation & validation for RAG retrieval evaluation.",
    no_args_is_help=True,
)
console = Console(soft_wrap=True)

_CONFIG_OPTION = typer.Option(..., "--config", exists=True, dir_okay=False, help="Run config YAML")
_DEFAULT_K = 10


@app.callback()
def _main(
    *,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),  # noqa: FBT003 - typer API
) -> None:
    """Configure logging for all commands."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def _load_or_exit(config_path: Path) -> dict[str, Any]:
    try:
        return load_config(config_path)
    except (ValueError, RegistryError) as error:
        console.print(f"[red]config error:[/red] {error}")
        raise typer.Exit(1) from error


@app.command()
def validate(config: Path = _CONFIG_OPTION) -> None:
    """Validate a config: schema, registry keys, judge-family rule."""
    loaded = _load_or_exit(config)
    warnings = validate_config(loaded)

    table = Table(title=f"ragsynth config: {loaded['ragsynth']['name']}")
    table.add_column("section")
    table.add_column("type / params")
    for key, block in loaded["resources"].items():
        if isinstance(block, dict) and "type" in block:
            table.add_row(f"resources.{key}", str(block["type"]))
    for step in loaded["pipeline"]:
        table.add_row("pipeline", f"{step['type']}  {step.get('params', {})}")
    console.print(table)
    for warning in warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    console.print("[green]config OK[/green]")


@app.command()
def run(config: Path = _CONFIG_OPTION) -> None:
    """Run the full pipeline and write the experiment outputs."""
    loaded = _load_or_exit(config)
    console.print(f"building resources for [bold]{loaded['ragsynth']['name']}[/bold] ...")
    resources = build_resources(loaded)
    pipeline = build_pipeline(loaded, resources)
    state = make_initial_state(loaded)
    pipeline.fit(resources)
    state = pipeline.run(state)

    out_dir = Path(loaded["artifacts_dir"]).parent
    console.print(
        f"accepted [bold]{len(state.accepted)}[/bold] records, "
        f"rejected {len(state.rejected)} "
        f"(gate pass rate {state.metrics.get('gate_pass_rate', 1.0):.2f})"
    )
    report = state.metrics.get("eval_report")
    if report is not None:
        table = Table(title="4-arm validation summary")
        for column in ("arm", "n", "KL", "wC2ST", "ESS/N", "tau", "tau_AP", "gates"):
            table.add_column(column)
        for arm, block in report["arms"].items():
            if block.get("skipped"):
                table.add_row(arm, str(block["n_records"]), *["-"] * 5, "skipped")
                continue
            table.add_row(
                arm,
                str(block["n_records"]),
                f"{block['fidelity']['kl']:.3f}",
                "-"
                if block["fidelity"]["wc2st_mean"] is None
                else f"{block['fidelity']['wc2st_mean']:.3f}",
                f"{block['efficiency']['ess_ratio']:.2f}",
                f"{block['validity']['tau']:.3f}",
                f"{block['validity']['tau_ap']:.3f}",
                "PASS" if block["gates_passed"] else "fail",
            )
        console.print(table)
    console.print(f"outputs under [bold]{out_dir}[/bold]")


@app.command()
def report(config: Path = _CONFIG_OPTION) -> None:
    """Re-render report.md and figures from a previous run's metrics.json."""
    loaded = _load_or_exit(config)
    out_dir = Path(loaded["artifacts_dir"]).parent
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        console.print(f"[red]missing metrics.json[/red] under {out_dir}")
        console.print("execute `ragsynth run` with this config first")
        raise typer.Exit(1)
    payload = json.loads(metrics_path.read_text())
    eval_report = EvalReport(**payload)
    k = next(
        (
            step.get("params", {}).get("k", _DEFAULT_K)
            for step in loaded["pipeline"]
            if step["type"] == "validator"
        ),
        _DEFAULT_K,
    )
    (out_dir / "report.md").write_text(render_markdown(eval_report, k=k))
    render_figures(eval_report, out_dir / "figures", eval_report.gates.get("tau", 0.9))
    console.print(f"re-rendered report.md and figures under [bold]{out_dir}[/bold]")


if __name__ == "__main__":
    app()
