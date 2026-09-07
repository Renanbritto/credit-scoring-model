"""
metrics.py
==========
Módulo para cálculo de métricas estatísticas e discriminatórias de risco de crédito:
Kolmogorov-Smirnov (K-S), Curva ROC / AUC, Coeficiente Gini e tabela de deciles de score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class KSResult:
    """Resultado da avaliação estatística Kolmogorov-Smirnov."""

    ks_stat: float  # Valor de K-S em percentual (ex: 48.6%)
    cutoff_at_max_ks: int  # Ponto de score onde ocorre a máxima separação
    table: pd.DataFrame  # Tabela com CDFs de Goods e Bads


@dataclass
class ModelPerformanceMetrics:
    """Consolidado de métricas estatísticas do modelo."""

    auc: float  # Area Under the ROC Curve (ex: 0.842)
    gini: float  # Coeficiente Gini (2 * AUC - 1)
    ks_stat: float  # Estatística K-S
    max_ks_score: int
    roc_curve_data: pd.DataFrame  # Dados para plotagem da curva ROC
    ks_curve_data: pd.DataFrame  # Dados para plotagem do K-S


def calculate_ks(
    scores: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 20,
) -> KSResult:
    """Calcula a estatística Kolmogorov-Smirnov (K-S) em faixas de score.

    K-S = max | CDF(Bads) - CDF(Goods) |

    Parâmetros
    ----------
    scores : np.ndarray
        Array com os scores dos proponentes (ex: 300 a 850).
    y_true : np.ndarray
        Array com a variável alvo (1 = Bad / Default, 0 = Good).
    n_bins : int
        Número de buckets/cortes para calcular a CDF acumulada.

    Retorno
    -------
    KSResult
        Objeto com a estatística K-S, corte de máxima separação e tabela detalhada.
    """
    df = pd.DataFrame({"score": scores, "bad": y_true, "good": 1 - y_true})
    df = df.sort_values(by="score", ascending=True).reset_index(drop=True)

    # Cria faixas discretas de score
    score_min, score_max = int(scores.min()), int(scores.max())
    bin_edges = np.linspace(score_min, score_max, n_bins + 1)

    df["bin"] = pd.cut(df["score"], bins=bin_edges, include_lowest=True)

    grouped = (
        df.groupby("bin", observed=False)
        .agg(
            total_count=("bad", "count"),
            bads_count=("bad", "sum"),
            goods_count=("good", "sum"),
            mean_score=("score", "mean"),
        )
        .reset_index()
    )

    grouped["empirical_bad_rate"] = np.where(
        grouped["total_count"] > 0,
        (grouped["bads_count"] / grouped["total_count"]) * 100.0,
        0.0,
    )

    # CDF acumulada de maus e bons
    total_bads = max(int(grouped["bads_count"].sum()), 1)
    total_goods = max(int(grouped["goods_count"].sum()), 1)

    grouped["cum_bads"] = grouped["bads_count"].cumsum()
    grouped["cum_goods"] = grouped["goods_count"].cumsum()

    grouped["cum_bads_pct"] = (grouped["cum_bads"] / total_bads) * 100.0
    grouped["cum_goods_pct"] = (grouped["cum_goods"] / total_goods) * 100.0

    # Diferença absoluta entre as curvas de distribuição acumulada
    grouped["ks_diff"] = grouped["cum_bads_pct"] - grouped["cum_goods_pct"]

    # Identifica o ponto de máxima divergência
    max_idx = grouped["ks_diff"].idxmax()
    max_ks = float(grouped.loc[max_idx, "ks_diff"])
    best_score = (
        round(grouped.loc[max_idx, "mean_score"])
        if not np.isnan(grouped.loc[max_idx, "mean_score"])
        else 540
    )

    return KSResult(
        ks_stat=round(max_ks, 2),
        cutoff_at_max_ks=best_score,
        table=grouped,
    )


def evaluate_model_performance(
    scores: np.ndarray,
    prob_bad: np.ndarray,
    y_true: np.ndarray,
) -> ModelPerformanceMetrics:
    """Calcula todas as métricas discriminatórias consolidadas."""
    # AUC
    auc_val = float(roc_auc_score(y_true, prob_bad))
    gini_val = float(2.0 * auc_val - 1.0)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, prob_bad)
    # Downsample para visualização ágil
    indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr))).astype(int)
    roc_df = pd.DataFrame(
        {
            "fpr": fpr[indices].round(4),
            "tpr": tpr[indices].round(4),
            "baseline": fpr[indices].round(4),
        }
    )

    # K-S
    ks_res = calculate_ks(scores, y_true)

    return ModelPerformanceMetrics(
        auc=round(auc_val, 4),
        gini=round(gini_val * 100.0, 2),  # em %
        ks_stat=ks_res.ks_stat,
        max_ks_score=ks_res.cutoff_at_max_ks,
        roc_curve_data=roc_df,
        ks_curve_data=ks_res.table,
    )
