#!/usr/bin/env python3
"""Gera somente os graficos de Gantt no estilo visual do material-base.

O script usa exclusivamente os CSVs detalhados em relatorio/dados_gantt.
Ele nao executa modelos de otimizacao e nao altera os dados de entrada.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parent
DADOS = ROOT / "dados_gantt"
FIGURAS = ROOT / "figuras" / "gantt"
TOL = 1.0e-6


@dataclass(frozen=True)
class Operacao:
    problema: str
    instancia: str
    job: str
    operation: str
    machine: str
    start: float
    finish: float
    duration: float
    makespan: float
    status_terminacao: str
    status_primal: str


def to_float(valor: str) -> float:
    return float((valor or "0").strip())


def read_gantt_csv(path: Path) -> list[Operacao]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        Operacao(
            problema=row["problema"],
            instancia=row["instancia"],
            job=row["job"],
            operation=row.get("operation", ""),
            machine=row["machine"],
            start=to_float(row["start"]),
            finish=to_float(row["finish"]),
            duration=to_float(row["duration"]),
            makespan=to_float(row["makespan"]),
            status_terminacao=row["status_terminacao"],
            status_primal=row["status_primal"],
        )
        for row in rows
    ]


def natural_key(text: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", text)
    if match:
        return text[: match.start()], int(match.group(1))
    return text, -1


def job_number(job: str) -> str:
    match = re.search(r"(\d+)$", job)
    return match.group(1) if match else job


def palette(keys: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return {key: cmap(i % cmap.N) for i, key in enumerate(keys)}


def validate_duration(rows: list[Operacao]) -> None:
    for row in rows:
        if not math.isclose(row.start + row.duration, row.finish, rel_tol=TOL, abs_tol=TOL):
            raise ValueError(
                f"Duracao inconsistente em {row.instancia}: "
                f"{row.job}-{row.operation} {row.machine}"
            )


def validate_no_machine_overlap(rows: list[Operacao]) -> None:
    by_machine: dict[str, list[Operacao]] = defaultdict(list)
    for row in rows:
        by_machine[row.machine].append(row)
    for machine, ops in by_machine.items():
        ordered = sorted(ops, key=lambda row: (row.start, row.finish, natural_key(row.job)))
        for prev, current in zip(ordered, ordered[1:]):
            if current.start < prev.finish - TOL:
                raise ValueError(
                    f"Sobreposicao na maquina {machine}: "
                    f"{prev.job}-{prev.operation} e {current.job}-{current.operation}"
                )


def validate_job_shop_precedence(rows: list[Operacao]) -> None:
    by_job: dict[str, list[Operacao]] = defaultdict(list)
    for row in rows:
        by_job[row.job].append(row)
    for job, ops in by_job.items():
        ordered = sorted(ops, key=lambda row: int(row.operation))
        for prev, current in zip(ordered, ordered[1:]):
            if current.start < prev.finish - TOL:
                raise ValueError(
                    f"Precedencia invalida no job {job}: "
                    f"O{prev.operation} termina depois de O{current.operation} iniciar"
                )


def validate_flow_shop_precedence(rows: list[Operacao]) -> None:
    by_job: dict[str, list[Operacao]] = defaultdict(list)
    for row in rows:
        by_job[row.job].append(row)
    for job, ops in by_job.items():
        ordered = sorted(ops, key=lambda row: natural_key(row.machine))
        for prev, current in zip(ordered, ordered[1:]):
            if current.start < prev.finish - TOL:
                raise ValueError(
                    f"Precedencia invalida no flow shop para {job}: "
                    f"{prev.machine} termina depois de {current.machine} iniciar"
                )


def validate_makespan(rows: list[Operacao]) -> None:
    max_finish = max(row.finish for row in rows)
    makespan = rows[0].makespan
    if not math.isclose(max_finish, makespan, rel_tol=TOL, abs_tol=TOL):
        raise ValueError(
            f"Makespan inconsistente em {rows[0].instancia}: "
            f"maior termino {max_finish}, makespan {makespan}"
        )
    for row in rows:
        if not math.isclose(row.makespan, makespan, rel_tol=TOL, abs_tol=TOL):
            raise ValueError(f"Makespan divergente em {row.instancia}: {row.job}")


def validate_all_rows_plotted(rows: list[Operacao], plotted: int) -> None:
    if plotted != len(rows):
        raise ValueError(
            f"Quantidade de barras inconsistente em {rows[0].instancia}: "
            f"{plotted} desenhadas, {len(rows)} linhas no CSV"
        )


def validate_rows(rows: list[Operacao], tipo: str, plotted: int) -> list[str]:
    validate_duration(rows)
    validate_no_machine_overlap(rows)
    if tipo == "job":
        validate_job_shop_precedence(rows)
    elif tipo == "flow":
        validate_flow_shop_precedence(rows)
    validate_makespan(rows)
    validate_all_rows_plotted(rows, plotted)
    return [
        "inicio + duracao corresponde ao termino",
        "nao ha sobreposicao indevida na mesma maquina",
        "precedencias validas para os dados disponiveis",
        "maior termino corresponde ao makespan",
        "todas as linhas do CSV foram desenhadas",
    ]


def setup_axes(fig_height: float, title: str) -> tuple[plt.Figure, plt.Axes]:
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(13.5, fig_height), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel("Tempo")
    ax.grid(axis="x", linestyle=":", color="#b8b8b8", alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def save_outputs(fig: plt.Figure, base: Path) -> tuple[Path, Path]:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def draw_bar(
    ax: plt.Axes,
    row: Operacao,
    y: float,
    height: float,
    color,
    label: str,
    fontsize: int = 9,
) -> None:
    width = row.finish - row.start
    text_fontsize = max(5, min(fontsize, int(width * 2.2 + 3)))
    rect = Rectangle(
        (row.start, y - height / 2),
        width,
        height,
        facecolor=color,
        edgecolor="#222222",
        linewidth=0.7,
        alpha=0.92,
    )
    ax.add_patch(rect)
    ax.text(
        row.start + width / 2,
        y,
        label,
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=text_fontsize,
        clip_on=True,
    )


def plot_machine(rows: list[Operacao]) -> tuple[Path, Path, list[str]]:
    instancia = rows[0].instancia
    makespan = rows[0].makespan
    machines = sorted({row.machine for row in rows}, key=natural_key)
    jobs = sorted({row.job for row in rows}, key=natural_key)
    colors = palette(jobs)
    ypos = {machine: i for i, machine in enumerate(machines)}
    fig, ax = setup_axes(2.6 + 0.45 * len(machines), f"Machine Scheduling - {instancia} - makespan = {makespan:.3f}")
    plotted = 0
    for row in sorted(rows, key=lambda item: (natural_key(item.machine), item.start, natural_key(item.job))):
        draw_bar(ax, row, ypos[row.machine], 0.48, colors[row.job], f"Tarefa {row.job}", fontsize=10)
        plotted += 1

    ax.set_ylabel("Maquina")
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(machines)
    ax.set_ylim(-0.75, len(machines) - 0.25)
    ax.set_xlim(-0.02 * makespan, makespan * 1.06)
    ax.invert_yaxis()
    handles = [Patch(facecolor=colors[job], edgecolor="#222222", label=f"Tarefa {job}") for job in jobs]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", title="Tarefas")

    validation = validate_rows(rows, "machine", plotted)
    return (*save_outputs(fig, FIGURAS / f"gantt_machine_{instancia}_estilo_professor"), validation)


def plot_job_shop(rows: list[Operacao]) -> tuple[Path, Path, list[str]]:
    instancia = rows[0].instancia
    makespan = rows[0].makespan
    machines = sorted({row.machine for row in rows}, key=natural_key)
    jobs = sorted({row.job for row in rows}, key=natural_key)
    colors = palette(jobs)
    ypos = {machine: i for i, machine in enumerate(machines)}
    fig, ax = setup_axes(2.8 + 0.48 * len(machines), f"Job Shop Scheduling - {instancia} - makespan = {makespan:.3f}")
    plotted = 0
    for row in sorted(rows, key=lambda item: (natural_key(item.machine), item.start, int(item.operation))):
        draw_bar(ax, row, ypos[row.machine], 0.55, colors[row.job], f"{row.job}-O{row.operation}", fontsize=8)
        plotted += 1

    ax.set_ylabel("Maquina")
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(machines)
    ax.set_ylim(-0.75, len(machines) - 0.25)
    ax.set_xlim(-0.02 * makespan, makespan * 1.06)
    ax.invert_yaxis()
    handles = [Patch(facecolor=colors[job], edgecolor="#222222", label=job) for job in jobs]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", title="Jobs")

    validation = validate_rows(rows, "job", plotted)
    return (*save_outputs(fig, FIGURAS / f"gantt_job_shop_{instancia}_estilo_professor"), validation)


def plot_flow_shop(rows: list[Operacao]) -> tuple[Path, Path, list[str]]:
    instancia = rows[0].instancia
    makespan = rows[0].makespan
    machines = sorted({row.machine for row in rows}, key=natural_key)
    jobs = sorted({row.job for row in rows}, key=natural_key)
    colors = palette(jobs)
    ypos = {machine: i for i, machine in enumerate(machines)}
    fig, ax = setup_axes(2.7 + 0.55 * len(machines), f"Flow Shop Scheduling - {instancia} - makespan = {makespan:.3f}")
    plotted = 0
    for row in sorted(rows, key=lambda item: (natural_key(item.machine), item.start, natural_key(item.job))):
        draw_bar(ax, row, ypos[row.machine], 0.55, colors[row.job], f"T{job_number(row.job)}", fontsize=9)
        plotted += 1

    ax.set_ylabel("Maquina")
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(machines)
    ax.set_ylim(-0.75, len(machines) - 0.25)
    ax.set_xlim(-0.02 * makespan, makespan * 1.06)
    ax.invert_yaxis()
    handles = [Patch(facecolor=colors[job], edgecolor="#222222", label=f"T{job_number(job)}") for job in jobs]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", title="Tarefas")

    validation = validate_rows(rows, "flow", plotted)
    return (*save_outputs(fig, FIGURAS / f"gantt_flow_shop_{instancia}_estilo_professor"), validation)


def line(label: str, value: str) -> None:
    print(f"{label}: {value}")


def main() -> None:
    specs = [
        ("Machine Scheduling", DADOS / "gantt_machine.csv", plot_machine),
        ("Job Shop Scheduling", DADOS / "gantt_job_shop.csv", plot_job_shop),
        ("Flow Shop Scheduling", DADOS / "gantt_flow_shop.csv", plot_flow_shop),
    ]
    print("GERACAO_GANTTS_ESTILO_PROFESSOR")
    for _problem, path, plotter in specs:
        rows = read_gantt_csv(path)
        png, pdf, validation = plotter(rows)
        line("problema", rows[0].problema)
        line("instancia", rows[0].instancia)
        line("csv", str(path))
        line("linhas_csv", str(len(rows)))
        line("makespan", f"{rows[0].makespan:.12g}")
        line("png", str(png))
        line("pdf", str(pdf))
        line("validacao", "; ".join(validation))
        print()


if __name__ == "__main__":
    main()
