"""
test_woe_iv.py
==============
Testes unitários para cálculo de Weight of Evidence (WoE) e Information Value (IV).
"""

import pytest

from src.data_generator import generate_credit_data
from src.woe_iv import WoETransformer, classify_predictive_power


@pytest.fixture
def sample_data():
    return generate_credit_data(n_samples=2000, seed=123)


def test_woe_transformer_fit_transform(sample_data):
    transformer = WoETransformer()
    transformer.fit(sample_data, target_col="default")

    assert len(transformer.features_iv) > 0
    assert "dti" in transformer.features_iv
    assert "delinquency_2y" in transformer.features_iv

    # Testa transformação
    df_woe = transformer.transform(sample_data)
    assert df_woe.shape[0] == sample_data.shape[0]
    assert "dti_woe" in df_woe.columns
    assert not df_woe.isnull().any().any()


def test_woe_iv_values_positive(sample_data):
    transformer = WoETransformer()
    transformer.fit(sample_data, target_col="default")

    for feat_id, res in transformer.features_iv.items():
        assert res.iv >= 0.0, f"IV para {feat_id} não pode ser negativo."
        assert len(res.bins) > 0
        total_dist_goods = sum(b.dist_goods for b in res.bins)
        total_dist_bads = sum(b.dist_bads for b in res.bins)
        assert pytest.approx(total_dist_goods, abs=1.0) == 100.0
        assert pytest.approx(total_dist_bads, abs=1.0) == 100.0


def test_classify_predictive_power():
    assert "Forte" in classify_predictive_power(0.35)
    assert "Médio" in classify_predictive_power(0.18)
    assert "Fraco" in classify_predictive_power(0.05)
    assert "Inútil" in classify_predictive_power(0.01)
