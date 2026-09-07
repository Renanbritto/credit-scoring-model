"""
data_generator.py
=================
Módulo para geração reprodutível de dados sintéticos de proponentes de crédito,
reproduzindo as distribuições estatísticas, correlações e taxas de inadimplência
observadas no mercado financeiro brasileiro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_credit_data(
    n_samples: int = 45000,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera um DataFrame com proponentes de crédito e variável alvo de inadimplência (default).

    Parâmetros
    ----------
    n_samples : int
        Número de proponentes a serem simulados (default: 45.000).
    seed : int
        Semente aleatória para reprodutibilidade.

    Retorno
    -------
    pd.DataFrame
        DataFrame contendo variáveis cadastrais, financeiras, comportamentais e bureau,
        além da variável alvo 'default' (1 = inadimplente / bad, 0 = adimplente / good).
    """
    rng = np.random.default_rng(seed)

    # 1. Variáveis Cadastrais e Sociodemográficas
    # Idade: distribuição normal truncada entre 18 e 75 anos
    age = np.clip(rng.normal(loc=38, scale=12, size=n_samples), 18, 75).astype(int)

    # Tempo de emprego atual em anos (exponencial com correlação leve com a idade)
    max_tenure_by_age = np.maximum(0.1, (age - 18) * 0.7)
    tenure_raw = rng.exponential(scale=3.8, size=n_samples)
    job_tenure = np.clip(np.minimum(tenure_raw, max_tenure_by_age), 0.1, 35.0).round(1)

    # Renda mensal bruta (Log-Normal com mediana ~ R$ 4.500)
    income = np.clip(
        np.exp(rng.normal(loc=8.4, scale=0.6, size=n_samples)), 1412.0, 65000.0
    ).round(2)

    # 2. Variáveis Financeiras e de Alavancagem
    # Comprometimento de Renda (DTI - Debt to Income em %): Beta distribuído entre 5% e 75%
    dti = (rng.beta(a=2.5, b=6.0, size=n_samples) * 100.0).round(1)
    dti = np.clip(dti, 3.0, 80.0)

    # Utilização do Limite Rotativo (%): Beta distribuído entre 2% e 98%
    revolving_util = (rng.beta(a=1.8, b=2.8, size=n_samples) * 100.0).round(1)
    revolving_util = np.clip(revolving_util, 1.0, 100.0)

    # 3. Variáveis Comportamentais e Bureau
    # Consultas ao Bureau de Crédito nos últimos 6 meses (Poisson)
    inquiries_lambda = 0.8 + 0.02 * dti + 0.015 * revolving_util
    bureau_inquiries = rng.poisson(lam=inquiries_lambda, size=n_samples)
    bureau_inquiries = np.clip(bureau_inquiries, 0, 12)

    # Atrasos de 30-59 dias nos últimos 24 meses (Zero-inflated Poisson)
    delinq_prob = 0.25 + 0.003 * revolving_util + 0.004 * dti - 0.01 * job_tenure
    delinq_prob = np.clip(delinq_prob, 0.05, 0.75)
    has_delinquency = rng.binomial(n=1, p=delinq_prob, size=n_samples)
    delinq_count = rng.poisson(lam=0.9, size=n_samples) + 1
    delinquency_2y = np.where(has_delinquency == 1, delinq_count, 0)
    delinquency_2y = np.clip(delinquency_2y, 0, 8)

    # Valor solicitado do empréstimo (R$)
    requested_amount = np.clip(
        income * rng.uniform(0.5, 3.5, size=n_samples),
        1000.0,
        50000.0,
    ).round(2)

    # 4. Cálculo da Probabilidade Latente de Default P(Y=1)
    # Log-odds com coeficientes econométricos realistas
    # Fatores de risco: DTI alto (+), Rotativo alto (+), Atrasos (+), Consultas (+)
    # Fatores protetivos: Tempo de emprego (-), Idade (-)
    z = (
        -2.95
        + 0.042 * (dti - 25.0)
        + 0.028 * (revolving_util - 40.0)
        + 0.650 * delinquency_2y
        + 0.280 * bureau_inquiries
        - 0.120 * job_tenure
        - 0.025 * (age - 35.0)
    )

    # Função Sigmóide: P(Default) = 1 / (1 + exp(-z))
    default_prob = 1.0 / (1.0 + np.exp(-z))
    default_prob = np.clip(default_prob, 0.005, 0.95)

    # Realização binomial do default
    default = rng.binomial(n=1, p=default_prob, size=n_samples)

    df = pd.DataFrame(
        {
            "applicant_id": [f"PROP-{i + 1:06d}" for i in range(n_samples)],
            "age": age,
            "job_tenure": job_tenure,
            "income": income,
            "requested_amount": requested_amount,
            "dti": dti,
            "revolving_util": revolving_util,
            "delinquency_2y": delinquency_2y,
            "bureau_inquiries": bureau_inquiries,
            "default": default,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_credit_data()
    print(
        f"Dataset gerado com sucesso: {df.shape[0]} proponentes e {df.shape[1]} colunas."
    )
    print(
        f"Taxa média de inadimplência (Bad Rate global): {df['default'].mean() * 100:.2f}%"
    )
