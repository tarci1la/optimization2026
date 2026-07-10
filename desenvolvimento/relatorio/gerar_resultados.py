#!/usr/bin/env python3
"""Gera tabelas, graficos e dashboard a partir dos resultados consolidados.

Entrada principal:
    desenvolvimento/resultados_completos.csv

Este script nao executa modelos de otimizacao e nao modifica os CSVs originais.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELATORIO = ROOT / "relatorio"
RESULTADOS = ROOT / "resultados_completos.csv"
TABELAS = RELATORIO / "tabelas"
FIGURAS = RELATORIO / "figuras"
DASHBOARD = RELATORIO / "dashboard"
GANTT_DADOS = RELATORIO / "dados_gantt"
GANTT_FIGS = FIGURAS / "gantt"

PROBLEMAS = ["Machine Scheduling", "Job Shop Scheduling", "Flow Shop Scheduling"]
GANTT_INSTANCIAS = {
    "Machine Scheduling": "inst_n05_s01",
    "Job Shop Scheduling": "ft06",
    "Flow Shop Scheduling": "problem_3m_10j",
}


def ensure_dirs() -> None:
    for path in [TABELAS, FIGURAS, DASHBOARD]:
        path.mkdir(parents=True, exist_ok=True)


def to_float(value: str) -> float:
    value = (value or "").strip()
    if not value:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def to_int(value: str) -> int:
    value = (value or "").strip()
    if not value:
        return 0
    return int(float(value))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["_tarefas"] = to_float(row.get("tarefas", ""))
        row["_maquinas"] = to_float(row.get("maquinas", ""))
        row["_operacoes"] = to_float(row.get("operacoes", ""))
        row["_valor"] = to_float(row.get("valor_objetivo_ou_makespan", ""))
        row["_gap"] = to_float(row.get("gap_relativo", ""))
        row["_tempo"] = to_float(row.get("tempo_solucao", ""))
        row["_variaveis"] = to_float(row.get("quantidade_variaveis", ""))
        row["_restricoes"] = to_float(row.get("quantidade_restricoes", ""))
    return rows


def clean(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def mean(values: list[float]) -> float:
    vals = clean(values)
    return sum(vals) / len(vals) if vals else math.nan


def median(values: list[float]) -> float:
    vals = sorted(clean(values))
    if not vals:
        return math.nan
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def fmt(value: float, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def safe_name(name: str) -> str:
    return (
        name.lower()
        .replace(" scheduling", "")
        .replace(" ", "_")
        .replace("/", "_")
    )


def group_by_problem(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {p: [] for p in PROBLEMAS}
    for row in rows:
        grouped.setdefault(row["problema"], []).append(row)
    return grouped


def status_counts(rows: list[dict[str, str]]) -> Counter:
    return Counter(row["status_terminacao"] for row in rows)


def general_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = group_by_problem(rows)
    summary: list[dict[str, str]] = []
    for problem in PROBLEMAS:
        group = grouped.get(problem, [])
        non_opt = [r for r in group if r["status_terminacao"] != "OPTIMAL"]
        summary.append(
            {
                "problema": problem,
                "instancias": str(len(group)),
                "otimas": str(sum(r["status_terminacao"] == "OPTIMAL" for r in group)),
                "viaveis_nao_otimas": str(
                    sum(
                        r["status_primal"] == "FEASIBLE_POINT"
                        and r["status_terminacao"] != "OPTIMAL"
                        and r["status_terminacao"] != "ERRO"
                        for r in group
                    )
                ),
                "limite_tempo": str(
                    sum(
                        r["status_terminacao"] == "TIME_LIMIT"
                        or r["limite_tempo"].lower() == "true"
                        for r in group
                    )
                ),
                "erros": str(
                    sum(
                        r["status_terminacao"] == "ERRO"
                        or bool(r.get("mensagem_erro", "").strip())
                        for r in group
                    )
                ),
                "tempo_medio_s": fmt(mean([r["_tempo"] for r in group])),
                "tempo_mediano_s": fmt(median([r["_tempo"] for r in group])),
                "maior_tempo_s": fmt(max(clean([r["_tempo"] for r in group]))),
                "gap_medio": fmt(mean([r["_gap"] for r in group]), 4),
                "maior_gap": fmt(max(clean([r["_gap"] for r in group])), 4),
                "media_tarefas": fmt(mean([r["_tarefas"] for r in group])),
                "media_maquinas": fmt(mean([r["_maquinas"] for r in group])),
                "media_variaveis": fmt(mean([r["_variaveis"] for r in group])),
                "media_restricoes": fmt(mean([r["_restricoes"] for r in group])),
                "gap_medio_nao_otimas": fmt(mean([r["_gap"] for r in non_opt]), 4),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def render_table_png(path: Path, rows: list[dict[str, str]], fields: list[str], title: str) -> None:
    fig_h = max(2.6, 0.38 * (len(rows) + 2))
    fig_w = max(10, 1.15 * len(fields))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if not rows:
        ax.text(0.5, 0.5, "Nenhum registro encontrado", ha="center", va="center", fontsize=12)
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return
    data = [[row.get(field, "") for field in fields] for row in rows]
    table = ax.table(cellText=data, colLabels=fields, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e9eef5")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def base_table_fields() -> list[str]:
    return [
        "problema",
        "instancia",
        "tarefas",
        "maquinas",
        "operacoes",
        "status_terminacao",
        "status_primal",
        "valor_objetivo_ou_makespan",
        "gap_relativo",
        "tempo_solucao",
        "quantidade_variaveis",
        "quantidade_restricoes",
    ]


def build_specific_tables(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = group_by_problem(rows)
    tables: dict[str, list[dict[str, str]]] = {}

    slow: list[dict[str, str]] = []
    gaps: list[dict[str, str]] = []
    large: list[dict[str, str]] = []
    for problem in PROBLEMAS:
        group = grouped.get(problem, [])
        slow.extend(sorted(group, key=lambda r: r["_tempo"], reverse=True)[:5])
        gaps.extend(sorted(group, key=lambda r: r["_gap"], reverse=True)[:5])
        large.extend(
            sorted(
                group,
                key=lambda r: (r["_operacoes"], r["_tarefas"], r["_maquinas"]),
                reverse=True,
            )[:5]
        )

    tables["cinco_mais_demoradas_por_problema"] = slow
    tables["limite_tempo"] = [
        r
        for r in rows
        if r["status_terminacao"] == "TIME_LIMIT" or r["limite_tempo"].lower() == "true"
    ]
    tables["maior_gap_por_problema"] = gaps
    tables["erros"] = [
        r
        for r in rows
        if r["status_terminacao"] == "ERRO" or bool(r.get("mensagem_erro", "").strip())
    ]
    tables["maior_porte_por_problema"] = large
    tables["instancias_gantt"] = [
        r for r in rows if GANTT_INSTANCIAS.get(r["problema"]) == r["instancia"]
    ]
    return tables


def save_all_tables(rows: list[dict[str, str]]) -> dict[str, Path]:
    created: dict[str, Path] = {}
    summary = general_summary(rows)
    summary_fields = list(summary[0].keys())
    path = TABELAS / "tabela_geral_por_problema.csv"
    write_csv(path, summary, summary_fields)
    render_table_png(
        TABELAS / "tabela_geral_por_problema.png",
        summary,
        summary_fields,
        "Tabela geral por problema",
    )
    created["tabela_geral_csv"] = path
    created["tabela_geral_png"] = TABELAS / "tabela_geral_por_problema.png"

    fields = base_table_fields()
    for name, table_rows in build_specific_tables(rows).items():
        path = TABELAS / f"{name}.csv"
        write_csv(path, table_rows, fields)
        render_table_png(TABELAS / f"{name}.png", table_rows, fields, name.replace("_", " "))
        created[f"{name}_csv"] = path
        created[f"{name}_png"] = TABELAS / f"{name}.png"
    return created


def readable_instance(row: dict[str, str]) -> str:
    return f"{row['problema'].replace(' Scheduling', '')}\\n{row['instancia']}"


def should_log_scale(values: list[float]) -> bool:
    vals = [v for v in clean(values) if v > 0]
    return bool(vals) and max(vals) / min(vals) > 100


def savefig(fig: plt.Figure, base: Path) -> tuple[Path, Path]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_time_by_instance(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    ordered = sorted(rows, key=lambda r: (r["problema"], r["_tempo"]))
    y = [r["_tempo"] for r in ordered]
    x = np.arange(len(ordered))
    colors = {
        "Machine Scheduling": "#3b6ea8",
        "Job Shop Scheduling": "#c44e52",
        "Flow Shop Scheduling": "#55a868",
    }
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.bar(x, y, color=[colors.get(r["problema"], "gray") for r in ordered])
    ax.set_title("Tempo de solucao por instancia")
    ax.set_xlabel("Instancia")
    ax.set_ylabel("Tempo de solucao (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["instancia"] for r in ordered], rotation=70, ha="right", fontsize=7)
    if should_log_scale(y):
        ax.set_yscale("log")
        ax.set_ylabel("Tempo de solucao (s, escala log)")
    handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=problem)
        for problem, color in colors.items()
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    return savefig(fig, FIGURAS / "tempo_solucao_por_instancia")


def plot_status_counts(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    grouped = group_by_problem(rows)
    statuses = sorted(set(r["status_terminacao"] for r in rows))
    x = np.arange(len(PROBLEMAS))
    bottom = np.zeros(len(PROBLEMAS))
    colors = {"OPTIMAL": "#4c9f70", "TIME_LIMIT": "#d9903d", "ERRO": "#c44e52"}
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for status in statuses:
        values = [sum(r["status_terminacao"] == status for r in grouped[p]) for p in PROBLEMAS]
        ax.bar(x, values, bottom=bottom, label=status, color=colors.get(status, "gray"))
        bottom += np.array(values)
    ax.set_title("Quantidade de instancias por status")
    ax.set_xlabel("Problema")
    ax.set_ylabel("Quantidade de instancias")
    ax.set_xticks(x)
    ax.set_xticklabels(PROBLEMAS, rotation=15, ha="right")
    ax.legend(title="Status")
    return savefig(fig, FIGURAS / "quantidade_instancias_por_status")


def plot_mean_time(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    grouped = group_by_problem(rows)
    means = [mean([r["_tempo"] for r in grouped[p]]) for p in PROBLEMAS]
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.bar(PROBLEMAS, means, color=["#3b6ea8", "#c44e52", "#55a868"])
    ax.set_title("Tempo medio de solucao por problema")
    ax.set_xlabel("Problema")
    ax.set_ylabel("Tempo medio de solucao (s)")
    ax.tick_params(axis="x", rotation=15)
    return savefig(fig, FIGURAS / "tempo_medio_por_problema")


def scatter_metric_time(rows: list[dict[str, str]], metric: str, xlabel: str, filename: str) -> tuple[Path, Path]:
    colors = {
        "Machine Scheduling": "#3b6ea8",
        "Job Shop Scheduling": "#c44e52",
        "Flow Shop Scheduling": "#55a868",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    for problem in PROBLEMAS:
        group = [r for r in rows if r["problema"] == problem]
        ax.scatter(
            [r[metric] for r in group],
            [r["_tempo"] for r in group],
            label=problem,
            s=45,
            alpha=0.85,
            color=colors.get(problem),
        )
    ax.set_title(f"{xlabel} versus tempo de solucao")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tempo de solucao (s)")
    if should_log_scale([r["_tempo"] for r in rows]):
        ax.set_yscale("log")
        ax.set_ylabel("Tempo de solucao (s, escala log)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    return savefig(fig, FIGURAS / filename)


def plot_gap_by_instance(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    ordered = sorted(rows, key=lambda r: (r["problema"], r["_gap"]))
    y = [r["_gap"] for r in ordered]
    x = np.arange(len(ordered))
    colors = {
        "Machine Scheduling": "#3b6ea8",
        "Job Shop Scheduling": "#c44e52",
        "Flow Shop Scheduling": "#55a868",
    }
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.bar(x, y, color=[colors.get(r["problema"], "gray") for r in ordered])
    ax.set_title("Gap relativo por instancia")
    ax.set_xlabel("Instancia")
    ax.set_ylabel("Gap relativo")
    ax.set_xticks(x)
    ax.set_xticklabels([r["instancia"] for r in ordered], rotation=70, ha="right", fontsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=problem)
        for problem, color in colors.items()
    ]
    ax.legend(handles=handles, loc="upper left")
    return savefig(fig, FIGURAS / "gap_por_instancia")


def plot_hardest_time(rows: list[dict[str, str]], n: int = 12) -> tuple[Path, Path]:
    hardest = sorted(rows, key=lambda r: r["_tempo"], reverse=True)[:n]
    x = np.arange(len(hardest))
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.barh(x, [r["_tempo"] for r in hardest], color="#7d5ba6")
    ax.set_yticks(x)
    ax.set_yticklabels([readable_instance(r) for r in hardest], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Instancias com maior tempo de solucao")
    ax.set_xlabel("Tempo de solucao (s)")
    ax.set_ylabel("Instancia")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    return savefig(fig, FIGURAS / "instancias_maior_tempo")


def save_all_figures(rows: list[dict[str, str]]) -> dict[str, tuple[Path, Path]]:
    figures = {
        "tempo_solucao_por_instancia": plot_time_by_instance(rows),
        "quantidade_instancias_por_status": plot_status_counts(rows),
        "tempo_medio_por_problema": plot_mean_time(rows),
        "tarefas_versus_tempo": scatter_metric_time(rows, "_tarefas", "Numero de tarefas", "tarefas_versus_tempo"),
        "variaveis_versus_tempo": scatter_metric_time(rows, "_variaveis", "Numero de variaveis", "variaveis_versus_tempo"),
        "restricoes_versus_tempo": scatter_metric_time(rows, "_restricoes", "Numero de restricoes", "restricoes_versus_tempo"),
        "gap_por_instancia": plot_gap_by_instance(rows),
        "instancias_maior_tempo": plot_hardest_time(rows),
    }
    return figures


def draw_cards(fig: plt.Figure, summary: dict[str, str]) -> None:
    cards = [
        ("Instancias", summary["instancias"]),
        ("Otimas", summary["otimas"]),
        ("Viaveis nao otimas", summary["viaveis_nao_otimas"]),
        ("Time limit", summary["limite_tempo"]),
        ("Erros", summary["erros"]),
        ("Tempo medio", f"{summary['tempo_medio_s']} s"),
        ("Maior tempo", f"{summary['maior_tempo_s']} s"),
        ("Gap medio", summary["gap_medio"]),
    ]
    for i, (label, value) in enumerate(cards):
        x = 0.025 + (i % 4) * 0.24
        y = 0.89 - (i // 4) * 0.095
        ax = fig.add_axes([x, y, 0.215, 0.075])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#f2f5f8", edgecolor="#b9c2cc", linewidth=1.0))
        ax.text(0.04, 0.65, label, fontsize=9, color="#4a5560")
        ax.text(0.04, 0.18, value, fontsize=16, fontweight="bold", color="#1f2933")


def small_status_axis(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    grouped = group_by_problem(rows)
    statuses = sorted(set(r["status_terminacao"] for r in rows))
    x = np.arange(len(PROBLEMAS))
    bottom = np.zeros(len(PROBLEMAS))
    colors = {"OPTIMAL": "#4c9f70", "TIME_LIMIT": "#d9903d", "ERRO": "#c44e52"}
    for status in statuses:
        values = [sum(r["status_terminacao"] == status for r in grouped[p]) for p in PROBLEMAS]
        ax.bar(x, values, bottom=bottom, label=status, color=colors.get(status, "gray"))
        bottom += np.array(values)
    ax.set_title("Status das execucoes", fontsize=10)
    ax.set_ylabel("Instancias")
    ax.set_xticks(x)
    ax.set_xticklabels(["Machine", "Job Shop", "Flow Shop"], rotation=10, ha="right", fontsize=8)
    ax.legend(fontsize=7)


def small_mean_time_axis(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    grouped = group_by_problem(rows)
    values = [mean([r["_tempo"] for r in grouped[p]]) for p in PROBLEMAS]
    ax.bar(["Machine", "Job Shop", "Flow Shop"], values, color=["#3b6ea8", "#c44e52", "#55a868"])
    ax.set_title("Tempo medio por problema", fontsize=10)
    ax.set_ylabel("Segundos")
    ax.tick_params(axis="x", rotation=10, labelsize=8)


def small_scatter_axis(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    colors = {
        "Machine Scheduling": "#3b6ea8",
        "Job Shop Scheduling": "#c44e52",
        "Flow Shop Scheduling": "#55a868",
    }
    for problem in PROBLEMAS:
        group = [r for r in rows if r["problema"] == problem]
        ax.scatter([r["_variaveis"] for r in group], [r["_tempo"] for r in group], label=problem.replace(" Scheduling", ""), s=20, color=colors[problem])
    ax.set_title("Variaveis versus tempo", fontsize=10)
    ax.set_xlabel("Variaveis")
    ax.set_ylabel("Tempo (s)")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=7)


def small_hard_table_axis(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    ax.axis("off")
    hardest = sorted(rows, key=lambda r: r["_tempo"], reverse=True)[:6]
    table_rows = [
        [r["problema"].replace(" Scheduling", ""), r["instancia"], fmt(r["_tempo"]), fmt(r["_gap"], 3)]
        for r in hardest
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Problema", "Instancia", "Tempo (s)", "Gap"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)
    ax.set_title("Instancias mais dificeis", fontsize=10)


def small_gantt_axis(ax: plt.Axes) -> None:
    path = GANTT_FIGS / "gantt_machine_inst_n05_s01.png"
    ax.axis("off")
    if path.exists():
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title("Gantt representativo: Machine Scheduling", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Gantt representativo nao encontrado", ha="center", va="center")


def save_dashboard(rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> tuple[Path, Path]:
    total = {
        "instancias": str(len(rows)),
        "otimas": str(sum(r["status_terminacao"] == "OPTIMAL" for r in rows)),
        "viaveis_nao_otimas": str(
            sum(r["status_primal"] == "FEASIBLE_POINT" and r["status_terminacao"] != "OPTIMAL" for r in rows)
        ),
        "limite_tempo": str(sum(r["status_terminacao"] == "TIME_LIMIT" or r["limite_tempo"].lower() == "true" for r in rows)),
        "erros": str(sum(r["status_terminacao"] == "ERRO" or bool(r.get("mensagem_erro", "").strip()) for r in rows)),
        "tempo_medio_s": fmt(mean([r["_tempo"] for r in rows])),
        "maior_tempo_s": fmt(max(clean([r["_tempo"] for r in rows]))),
        "gap_medio": fmt(mean([r["_gap"] for r in rows]), 4),
    }
    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    fig.suptitle("Dashboard dos experimentos computacionais", fontsize=18, fontweight="bold", y=0.985)
    draw_cards(fig, total)

    ax1 = fig.add_axes([0.05, 0.54, 0.28, 0.24])
    ax2 = fig.add_axes([0.37, 0.54, 0.26, 0.24])
    ax3 = fig.add_axes([0.68, 0.54, 0.27, 0.24])
    ax4 = fig.add_axes([0.05, 0.08, 0.43, 0.36])
    ax5 = fig.add_axes([0.54, 0.08, 0.41, 0.36])

    small_status_axis(ax1, rows)
    small_mean_time_axis(ax2, rows)
    small_scatter_axis(ax3, rows)
    small_hard_table_axis(ax4, rows)
    small_gantt_axis(ax5)

    png = DASHBOARD / "dashboard_resultados.png"
    pdf = DASHBOARD / "dashboard_resultados.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    ensure_dirs()
    rows = read_rows(RESULTADOS)
    summary = general_summary(rows)
    tables = save_all_tables(rows)
    figures = save_all_figures(rows)
    dashboard_png, dashboard_pdf = save_dashboard(rows, summary)

    print("ESTATISTICAS_UTILIZADAS")
    for row in summary:
        print(row)
    print("ARQUIVOS_TABELAS")
    for path in tables.values():
        print(path)
    print("ARQUIVOS_GRAFICOS")
    for png, pdf in figures.values():
        print(png)
        print(pdf)
    print("DASHBOARD")
    print(dashboard_png)
    print(dashboard_pdf)


if __name__ == "__main__":
    main()
