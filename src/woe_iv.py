"""
woe_iv.py
=========
Módulo de cálculo de Weight of Evidence (WoE) e Information Value (IV),
binning monotônico e transformação de atributos para modelagem de risco de crédito.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BinStatistics:
    """Estatísticas calculadas para um bin/faixa de um atributo."""

    bin_label: str
    total: int
    goods: int
    bads: int
    bad_rate: float  # %
    dist_goods: float  # %
    dist_bads: float  # %
    woe: float  # Weight of Evidence
    iv_contribution: float  # Contribuição do bin para o Information Value


@dataclass
class FeatureIVResult:
    """Resultado consolidado de WoE e IV para uma característica."""

    feature_id: str
    feature_name: str
    category: str  # "financeiro" | "comportamental" | "cadastral" | "bureau"
    iv: float  # Total Information Value
    predictive_power: str  # "Muito Forte" | "Forte" | "Médio" | "Fraco" | "Inútil"
    description: str
    bins: list[BinStatistics]


# Definição padrão das faixas de corte e regras de binning para cada variável
DEFAULT_BIN_CONFIGS = {
    "dti": {
        "name": "Comprometimento de Renda (DTI)",
        "category": "financeiro",
        "description": "Relação percentual entre o total de dívidas ativas mensais e a renda bruta comprovada.",
        "bins": [-np.inf, 15.0, 25.0, 35.0, 45.0, np.inf],
        "labels": ["< 15%", "15% - 25%", "25% - 35%", "35% - 45%", ">= 45%"],
    },
    "delinquency_2y": {
        "name": "Atrasos 30-59 Dias (Últimos 24m)",
        "category": "comportamental",
        "description": "Frequência de episódios de inadimplência leve registrados no histórico recente.",
        "bins": [-np.inf, 0.5, 1.5, 2.5, np.inf],
        "labels": [
            "0 ocorrências",
            "1 ocorrência",
            "2 ocorrências",
            ">= 3 ocorrências",
        ],
    },
    "revolving_util": {
        "name": "Utilização do Rotativo (%)",
        "category": "financeiro",
        "description": "Percentual de utilização do limite de crédito rotativo e cartões no momento da consulta.",
        "bins": [-np.inf, 20.0, 50.0, 75.0, np.inf],
        "labels": ["< 20%", "20% - 50%", "50% - 75%", ">= 75%"],
    },
    "job_tenure": {
        "name": "Tempo no Emprego Atual (Anos)",
        "category": "cadastral",
        "description": "Estabilidade profissional e tempo contínuo de vínculo empregatício comprovado.",
        "bins": [-np.inf, 1.0, 3.0, 7.0, np.inf],
        "labels": ["< 1 ano", "1 - 3 anos", "3 - 7 anos", ">= 7 anos"],
    },
    "bureau_inquiries": {
        "name": "Consultas ao Bureau (6 meses)",
        "category": "bureau",
        "description": "Sinal de busca recente e intensa por crédito no mercado em diferentes instituições.",
        "bins": [-np.inf, 0.5, 2.5, 5.5, np.inf],
        "labels": [
            "0 consultas",
            "1 - 2 consultas",
            "3 - 5 consultas",
            ">= 6 consultas",
        ],
    },
    "age": {
        "name": "Faixa Etária (Idade)",
        "category": "cadastral",
        "description": "Maturidade financeira e ciclo de vida do tomador de crédito.",
        "bins": [-np.inf, 25.0, 35.0, 50.0, np.inf],
        "labels": ["18 - 25 anos", "26 - 35 anos", "36 - 50 anos", "> 50 anos"],
    },
}


def classify_predictive_power(iv: float) -> str:
    """Classifica a força preditiva segundo o padrão clássico de Siddiqi (Credit Risk Scorecards)."""
    if iv > 0.50:
        return "Muito Forte (Revisar Superexposição)"
    if iv >= 0.30:
        return "Forte"
    if iv >= 0.10:
        return "Médio"
    if iv >= 0.02:
        return "Fraco"
    return "Inútil / Desprezível"


class WoETransformer:
    """Transformador estatístico para WoE e Information Value."""

    def __init__(self, bin_configs: dict | None = None):
        self.bin_configs = bin_configs or DEFAULT_BIN_CONFIGS
        self.features_iv: dict[str, FeatureIVResult] = {}
        self.woe_mappings: dict[str, dict[str, float]] = {}
        self.total_goods: int = 0
        self.total_bads: int = 0

    def fit(self, df: pd.DataFrame, target_col: str = "default") -> WoETransformer:
        """Calcula as tabelas de contingência, WoE e IV para todas as features configuradas.

        WoE = ln( (% Goods_i) / (% Bads_i) )
        IV = sum( (% Goods_i - % Bads_i) * WoE_i )
        """
        y = df[target_col].values
        self.total_bads = int(np.sum(y == 1))
        self.total_goods = int(np.sum(y == 0))

        if self.total_bads == 0 or self.total_goods == 0:
            raise ValueError(
                "O dataset deve conter pelo menos um bom e um mau pagador."
            )

        self.features_iv.clear()
        self.woe_mappings.clear()

        for feat_id, config in self.bin_configs.items():
            if feat_id not in df.columns:
                continue

            # Realiza o binning categórico
            binned_series = pd.cut(
                df[feat_id],
                bins=config["bins"],
                labels=config["labels"],
                right=False,
            )

            bin_stats_list: list[BinStatistics] = []
            mapping_dict: dict[str, float] = {}
            total_iv = 0.0

            for label in config["labels"]:
                mask = binned_series == label
                sub_y = y[mask]
                total_in_bin = len(sub_y)
                bads_in_bin = int(np.sum(sub_y == 1))
                goods_in_bin = int(np.sum(sub_y == 0))

                # Suavização laplaciana para evitar log(0) ou divisão por zero
                goods_rate = max(goods_in_bin, 0.5) / self.total_goods
                bads_rate = max(bads_in_bin, 0.5) / self.total_bads

                woe = float(np.log(goods_rate / bads_rate))
                iv_contrib = float((goods_rate - bads_rate) * woe)
                total_iv += iv_contrib

                bad_rate_pct = (
                    (bads_in_bin / total_in_bin * 100.0) if total_in_bin > 0 else 0.0
                )
                dist_goods_pct = goods_in_bin / self.total_goods * 100.0
                dist_bads_pct = bads_in_bin / self.total_bads * 100.0

                stat = BinStatistics(
                    bin_label=str(label),
                    total=total_in_bin,
                    goods=goods_in_bin,
                    bads=bads_in_bin,
                    bad_rate=round(bad_rate_pct, 2),
                    dist_goods=round(dist_goods_pct, 2),
                    dist_bads=round(dist_bads_pct, 2),
                    woe=round(woe, 4),
                    iv_contribution=round(iv_contrib, 4),
                )
                bin_stats_list.append(stat)
                mapping_dict[str(label)] = woe

            iv_rounded = round(total_iv, 4)
            self.features_iv[feat_id] = FeatureIVResult(
                feature_id=feat_id,
                feature_name=config["name"],
                category=config["category"],
                iv=iv_rounded,
                predictive_power=classify_predictive_power(iv_rounded),
                description=config["description"],
                bins=bin_stats_list,
            )
            self.woe_mappings[feat_id] = mapping_dict

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte as variáveis contínuas em seus respectivos valores de WoE."""
        df_woe = pd.DataFrame(index=df.index)
        for feat_id, config in self.bin_configs.items():
            if feat_id not in df.columns:
                continue

            binned = pd.cut(
                df[feat_id],
                bins=config["bins"],
                labels=config["labels"],
                right=False,
            ).astype(str)

            mapping = self.woe_mappings.get(feat_id, {})
            df_woe[f"{feat_id}_woe"] = binned.map(mapping).fillna(0.0)

        return df_woe

    def get_summary_table(self) -> pd.DataFrame:
        """Retorna uma tabela síntese com o IV de cada variável ordenada por poder preditivo."""
        rows = []
        for feat_id, res in self.features_iv.items():
            rows.append(
                {
                    "Feature ID": feat_id,
                    "Variável": res.feature_name,
                    "Categoria": res.category.capitalize(),
                    "Information Value (IV)": res.iv,
                    "Poder Preditivo": res.predictive_power,
                    "Descrição": res.description,
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(by="Information Value (IV)", ascending=False)
            .reset_index(drop=True)
        )
