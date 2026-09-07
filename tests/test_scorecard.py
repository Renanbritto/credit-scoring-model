"""
test_scorecard.py
=================
Testes unitários para calibração econométrica FICO e pontuação.
"""

import numpy as np
import pytest

from src.data_generator import generate_credit_data
from src.scorecard import ScorecardModel


@pytest.fixture
def sample_data():
    return generate_credit_data(n_samples=2500, seed=42)


def test_scorecard_fit_and_predict(sample_data):
    model = ScorecardModel(target_score=600, target_odds=50, pdo=20)
    model.fit(sample_data, target_col="default")

    scores = model.predict_score(sample_data)
    pd_probs = model.predict_proba_default(sample_data)

    assert len(scores) == len(sample_data)
    assert len(pd_probs) == len(sample_data)

    # Scores devem estar estritamente dentro dos limites FICO [300, 850]
    assert (scores >= 300).all()
    assert (scores <= 850).all()

    # Relação inversa esperada: maior score deve ter menor probabilidade de default
    corr = np.corrcoef(scores, pd_probs)[0, 1]
    assert corr < -0.85, (
        f"Correlação Score vs PD deve ser fortemente negativa, obtido {corr}."
    )


def test_scorecard_rules_table(sample_data):
    model = ScorecardModel()
    model.fit(sample_data, target_col="default")

    df_rules = model.get_scorecard_dataframe()
    assert not df_rules.empty
    assert "Pontos Scorecard" in df_rules.columns
    assert "WoE" in df_rules.columns
