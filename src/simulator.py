"""
simulator.py
============
Módulo de Simulação Tática de Concessão de Crédito, Trade-off de Cut-off,
Cálculo de Perda Esperada (EL = PD x LGD x EAD) e Projeção Financeira de Margem Líquida.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """Resultado da simulação financeira para uma nota de corte (Cut-off)."""

    cutoff_score: int
    total_applicants: int
    approved_count: int
    rejected_count: int
    approval_rate_pct: float
    approved_capital_brl: float  # Em R$
    approved_capital_million: float  # Em R$ Milhões
    bad_rate_approved_pct: float  # Inadimplência na safra aprovada
    bads_approved_count: int
    expected_loss_brl: float  # Perda Esperada (EL)
    expected_loss_million: float
    gross_interest_revenue_brl: float  # Receita Bruta de Juros Anual
    funding_cost_brl: float  # Custo de Captação (CDI)
    net_credit_margin_brl: float  # Margem Líquida de Crédito
    net_credit_margin_million: float
    net_credit_margin_pct: float  # Margem Líquida % sobre Receita
    bads_prevented_count: int  # Inadimplentes barrados
    loss_prevented_million: float  # Prejuízo evitado em R$ Milhões


class CreditPolicySimulator:
    """Simulador de Políticas de Risco e Ponto de Corte (Cut-off)."""

    def __init__(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        loan_amounts: np.ndarray | None = None,
    ):
        self.scores = np.asarray(scores)
        self.y_true = np.asarray(y_true)
        self.n_samples = len(self.scores)
        self.loan_amounts = (
            np.asarray(loan_amounts)
            if loan_amounts is not None
            else np.full(self.n_samples, 5000.0)
        )

    def simulate(
        self,
        cutoff_score: int = 540,
        application_volume: int | None = None,
        avg_ticket: float | None = None,
        interest_rate_monthly_pct: float = 3.2,
        lgd_pct: float = 65.0,
        funding_cost_annual_pct: float = 11.5,
    ) -> SimulationResult:
        """Executa a simulação para um determinado ponto de corte."""
        total_volume = application_volume or self.n_samples

        # Máscara empírica de aprovação na base histórica
        approved_mask = self.scores >= cutoff_score
        emp_approved_count = int(np.sum(approved_mask))
        emp_approval_rate = emp_approved_count / max(self.n_samples, 1)

        # Projeção sobre o volume pretendido
        projected_approved_count = round(total_volume * emp_approval_rate)
        projected_rejected_count = total_volume - projected_approved_count

        # Bad Rate empírica observada no grupo aprovado
        if emp_approved_count > 0:
            bads_in_approved = int(np.sum((approved_mask) & (self.y_true == 1)))
            bad_rate_approved = (bads_in_approved / emp_approved_count) * 100.0
        else:
            bad_rate_approved = 0.0

        projected_bads_approved = round(projected_approved_count * (bad_rate_approved / 100.0))

        # Ticket médio e capital
        ticket = (
            avg_ticket if avg_ticket is not None else float(np.mean(self.loan_amounts))
        )
        approved_capital = projected_approved_count * ticket
        approved_capital_m = approved_capital / 1_000_000.0

        # Perda Esperada (EL = Volume Aprovado * Bad Rate * LGD)
        expected_loss = (
            approved_capital * (bad_rate_approved / 100.0) * (lgd_pct / 100.0)
        )
        expected_loss_m = expected_loss / 1_000_000.0

        # Receita Bruta de Juros estimada para horizonte de 12 meses
        annual_interest_rate = ((1.0 + interest_rate_monthly_pct / 100.0) ** 12) - 1.0
        gross_revenue = approved_capital * annual_interest_rate

        # Custo de Captação (Funding / CDI)
        funding_cost = approved_capital * (funding_cost_annual_pct / 100.0)

        # Margem de Contribuição Líquida
        net_margin = gross_revenue - funding_cost - expected_loss
        net_margin_m = net_margin / 1_000_000.0
        net_margin_pct = (
            (net_margin / gross_revenue * 100.0) if gross_revenue > 0 else 0.0
        )

        # Inadimplentes recusados e perda evitada
        rejected_mask = ~approved_mask
        emp_rejected_count = int(np.sum(rejected_mask))
        if emp_rejected_count > 0:
            bads_in_rejected = int(np.sum(rejected_mask & (self.y_true == 1)))
            bad_rate_rejected = (bads_in_rejected / emp_rejected_count) * 100.0
        else:
            bad_rate_rejected = 0.0

        projected_bads_prevented = round(projected_rejected_count * (bad_rate_rejected / 100.0))
        loss_prevented_m = (
            projected_rejected_count
            * ticket
            * (bad_rate_rejected / 100.0)
            * (lgd_pct / 100.0)
        ) / 1_000_000.0

        return SimulationResult(
            cutoff_score=cutoff_score,
            total_applicants=total_volume,
            approved_count=projected_approved_count,
            rejected_count=projected_rejected_count,
            approval_rate_pct=round(emp_approval_rate * 100.0, 2),
            approved_capital_brl=approved_capital,
            approved_capital_million=round(approved_capital_m, 2),
            bad_rate_approved_pct=round(bad_rate_approved, 2),
            bads_approved_count=projected_bads_approved,
            expected_loss_brl=expected_loss,
            expected_loss_million=round(expected_loss_m, 2),
            gross_interest_revenue_brl=gross_revenue,
            funding_cost_brl=funding_cost,
            net_credit_margin_brl=net_margin,
            net_credit_margin_million=round(net_margin_m, 2),
            net_credit_margin_pct=round(net_margin_pct, 2),
            bads_prevented_count=projected_bads_prevented,
            loss_prevented_million=round(loss_prevented_m, 2),
        )

    def generate_cutoff_frontier(
        self,
        min_cutoff: int = 340,
        max_cutoff: int = 760,
        step: int = 20,
        application_volume: int = 10000,
        avg_ticket: float = 5000.0,
    ) -> pd.DataFrame:
        """Gera a curva de trade-off para múltiplos pontos de corte."""
        records = []
        for cut in range(min_cutoff, max_cutoff + 1, step):
            res = self.simulate(
                cutoff_score=cut,
                application_volume=application_volume,
                avg_ticket=avg_ticket,
            )
            records.append(
                {
                    "Cutoff Score": res.cutoff_score,
                    "Taxa Aprovação (%)": res.approval_rate_pct,
                    "Bad Rate (%)": res.bad_rate_approved_pct,
                    "Capital Aprovado (R$ M)": res.approved_capital_million,
                    "Perda Esperada (R$ M)": res.expected_loss_million,
                    "Margem Líquida (R$ M)": res.net_credit_margin_million,
                    "Margem Líquida (%)": res.net_credit_margin_pct,
                    "Perdas Evitadas (R$ M)": res.loss_prevented_million,
                }
            )
        return pd.DataFrame(records)
