"""
test_simulator.py
=================
Testes unitários para simulação de corte (Cut-off), Perda Esperada e Margem Líquida.
"""

import numpy as np
import pytest

from src.simulator import CreditPolicySimulator


@pytest.fixture
def simulator_instance():
    np.random.seed(42)
    scores = np.random.randint(320, 820, size=2000)
    # Quanto maior o score, menor a probabilidade de bad
    prob_bad = np.clip(0.50 - 0.0006 * (scores - 300), 0.01, 0.90)
    y_true = np.random.binomial(n=1, p=prob_bad)
    return CreditPolicySimulator(scores=scores, y_true=y_true)


def test_simulate_monotonicity(simulator_instance):
    # Ao aumentar a nota de corte, a taxa de aprovação DEVE cair e a qualidade (menor Bad Rate) DEVE melhorar
    res_low = simulator_instance.simulate(cutoff_score=400)
    res_high = simulator_instance.simulate(cutoff_score=650)

    assert res_low.approval_rate_pct > res_high.approval_rate_pct
    assert res_low.bad_rate_approved_pct > res_high.bad_rate_approved_pct
    assert res_high.loss_prevented_million >= res_low.loss_prevented_million


def test_generate_cutoff_frontier(simulator_instance):
    frontier = simulator_instance.generate_cutoff_frontier(
        min_cutoff=400, max_cutoff=700, step=50
    )
    assert not frontier.empty
    assert "Taxa Aprovação (%)" in frontier.columns
    assert "Margem Líquida (R$ M)" in frontier.columns
