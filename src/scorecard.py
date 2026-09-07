"""
scorecard.py
============
Módulo de Calibração Econométrica do Scorecard FICO (300-850) a partir de
Regressão Logística ponderada e scaling por Points to Double the Odds (PDO).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.woe_iv import WoETransformer


@dataclass
class ScorecardBinRule:
    """Regra de pontuação para um bin específico de um atributo."""

    feature_id: str
    feature_name: str
    category: str
    bin_label: str
    woe: float
    coefficient: float
    score_points: int


class ScorecardModel:
    """Modelo de Risco de Crédito baseado em Scorecard FICO (300-850).

    Parâmetros
    ----------
    target_score : float
        Score de referência (default: 600 pontos).
    target_odds : float
        Odds de bons/maus no score de referência (default: 50.0, ou seja, 50 Goods para 1 Bad).
    pdo : float
        Points to Double the Odds (default: 20.0 pontos).
    min_score : int
        Score mínimo da régua (default: 300).
    max_score : int
        Score máximo da régua (default: 850).
    """

    def __init__(
        self,
        target_score: float = 600.0,
        target_odds: float = 50.0,
        pdo: float = 20.0,
        min_score: int = 300,
        max_score: int = 850,
        regularization_c: float = 1.0,
    ):
        self.target_score = target_score
        self.target_odds = target_odds
        self.pdo = pdo
        self.min_score = min_score
        self.max_score = max_score
        self.regularization_c = regularization_c

        # Parâmetros de Scaling FICO
        self.factor: float = self.pdo / np.log(2.0)
        self.offset: float = self.target_score - self.factor * np.log(self.target_odds)

        self.lr_model: LogisticRegression | None = None
        self.woe_transformer: WoETransformer | None = None
        self.feature_names: list[str] = []
        self.coefficients: dict[str, float] = {}
        self.intercept: float = 0.0
        self.rules_table: list[ScorecardBinRule] = []

    def fit(self, df_raw: pd.DataFrame, target_col: str = "default") -> ScorecardModel:
        """Ajusta o transformador WoE e a Regressão Logística para calibrar o scorecard."""
        self.woe_transformer = WoETransformer()
        self.woe_transformer.fit(df_raw, target_col=target_col)

        # Matriz WoE
        df_woe = self.woe_transformer.transform(df_raw)
        self.feature_names = list(df_woe.columns)
        y = df_raw[target_col].values

        # Nota: y=1 é Bad, y=0 é Good.
        # WoE foi calculado como ln(Goods/Bads). Assim, valores altos de WoE indicam maior proporção de Goods.
        # Treinamos a regressão logística para prever Y=1 (Default):
        # logit(p_bad) = alpha + sum(beta_j * WoE_j)
        # Esperamos que beta_j < 0, pois maior WoE (mais Goods) reduz a probabilidade de default.
        self.lr_model = LogisticRegression(
            C=self.regularization_c,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self.lr_model.fit(df_woe, y)

        self.intercept = float(self.lr_model.intercept_[0])
        self.coefficients = {
            col: float(coef)
            for col, coef in zip(self.feature_names, self.lr_model.coef_[0])
        }

        # Constrói a tabela de pontos do Scorecard por bin
        self._build_rules_table()
        return self

    def _build_rules_table(self) -> None:
        """Gera a pontuação atribuída a cada faixa (bin) das variáveis."""
        if not self.woe_transformer:
            return

        self.rules_table.clear()
        n_features = len(self.feature_names)
        base_points_per_feature = (self.offset - self.factor * self.intercept) / max(
            n_features, 1
        )

        for col in self.feature_names:
            feat_id = col.replace("_woe", "")
            iv_res = self.woe_transformer.features_iv.get(feat_id)
            if not iv_res:
                continue

            beta = self.coefficients.get(col, 0.0)

            for b in iv_res.bins:
                # Como beta < 0 para WoE positivo em default:
                # Pontos = base_points - factor * (beta * WoE)
                # Assim, maior WoE resulta em acréscimo de pontuação!
                points = int(
                    np.round(base_points_per_feature - (self.factor * beta * b.woe))
                )

                rule = ScorecardBinRule(
                    feature_id=feat_id,
                    feature_name=iv_res.feature_name,
                    category=iv_res.category,
                    bin_label=b.bin_label,
                    woe=b.woe,
                    coefficient=round(beta, 4),
                    score_points=points,
                )
                self.rules_table.append(rule)

    def get_scorecard_dataframe(self) -> pd.DataFrame:
        """Retorna a tabela completa de regras do Scorecard para documentação e auditoria."""
        rows = []
        for r in self.rules_table:
            rows.append(
                {
                    "Feature": r.feature_name,
                    "Categoria": r.category.capitalize(),
                    "Faixa / Bin": r.bin_label,
                    "WoE": r.woe,
                    "Coeficiente": r.coefficient,
                    "Pontos Scorecard": r.score_points,
                }
            )
        return pd.DataFrame(rows)

    def predict_score(self, df_raw: pd.DataFrame) -> pd.Series:
        """Calcula o Score individual de cada proponente na escala [300, 850]."""
        if not self.woe_transformer or not self.lr_model:
            raise ValueError("O modelo precisa ser ajustado com fit() antes de prever.")

        # Obtém matriz WoE
        df_woe = self.woe_transformer.transform(df_raw)

        # Probabilidade prevista de default p = P(Y=1)
        prob_bad = self.lr_model.predict_proba(df_woe)[:, 1]
        prob_bad = np.clip(prob_bad, 1e-6, 1.0 - 1e-6)

        # Odds de adimplência: Odds = (1 - p) / p = P(Good) / P(Bad)
        odds_good = (1.0 - prob_bad) / prob_bad

        # Score = Offset + Factor * ln(Odds_good)
        scores = self.offset + self.factor * np.log(odds_good)
        scores_clipped = np.clip(
            np.round(scores), self.min_score, self.max_score
        ).astype(int)

        return pd.Series(scores_clipped, index=df_raw.index, name="credit_score")

    def predict_proba_default(self, df_raw: pd.DataFrame) -> pd.Series:
        """Calcula a Probabilidade de Default (PD) calibrada de cada proponente."""
        if not self.woe_transformer or not self.lr_model:
            raise ValueError("O modelo precisa ser ajustado com fit() antes de prever.")

        df_woe = self.woe_transformer.transform(df_raw)
        prob_bad = self.lr_model.predict_proba(df_woe)[:, 1]
        return pd.Series(prob_bad, index=df_raw.index, name="pd")
