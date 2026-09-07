"""
app.py
======
Dashboard Executivo Interativo de Modelagem Estatística de Risco de Crédito & Credit Scoring.
Desenvolvido por Renan Nocelli | Analista de Dados & Engenharia de Risco Financeiro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import generate_credit_data
from src.metrics import evaluate_model_performance
from src.scorecard import ScorecardModel
from src.simulator import CreditPolicySimulator

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA & DESIGN SYSTEM (DARK EXECUTIVE THEME)
# ==============================================================================
st.set_page_config(
    page_title="Credit Scoring & Risco de Crédito | Renan Nocelli",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Estilização Geral */
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-approved {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-rejected {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# CARREGAMENTO EM CACHE DOS DADOS E MODELO
# ==============================================================================
@st.cache_resource(show_spinner="Carregando e treinando pipeline de Scorecard FICO...")
def load_trained_system():
    # Carrega dados
    df = generate_credit_data(n_samples=45000, seed=42)
    model = ScorecardModel(target_score=600.0, target_odds=50.0, pdo=20.0)
    model.fit(df, target_col="default")

    scores = model.predict_score(df).values
    probs = model.predict_proba_default(df).values
    y_true = df["default"].values

    metrics = evaluate_model_performance(scores, probs, y_true)
    simulator = CreditPolicySimulator(
        scores=scores, y_true=y_true, loan_amounts=df["requested_amount"].values
    )

    return df, model, metrics, simulator


df_raw, scorecard_model, perf_metrics, sim_engine = load_trained_system()


# ==============================================================================
# BARRA LATERAL (CONTROLES DO SIMULADOR TÁTICO)
# ==============================================================================
with st.sidebar:
    st.image(
        "https://img.shields.io/badge/Model-FICO%20Scorecard-10b981?style=for-the-badge",
        width=220,
    )
    st.title("🎛️ Parâmetros da Esteira")
    st.markdown(
        "Configure as premissas da política de crédito e simule o impacto financeiro em tempo real."
    )

    cutoff_input = st.slider(
        "Ponto de Corte (Cut-off)",
        min_value=320,
        max_value=760,
        value=540,
        step=10,
        help="Proponentes com Score igual ou superior a esta nota serão APROVADOS.",
    )

    st.markdown("---")
    st.subheader("💼 Premissas da Carteira")
    vol_input = st.number_input(
        "Volume de Propostas / Mês",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
    )
    ticket_input = st.number_input(
        "Tíquete Médio (R$)",
        min_value=1000.0,
        max_value=100000.0,
        value=5000.0,
        step=500.0,
    )
    interest_monthly = st.slider(
        "Taxa de Juros (% a.m.)",
        min_value=0.5,
        max_value=10.0,
        value=3.2,
        step=0.1,
    )
    lgd_input = st.slider(
        "LGD - Perda Dado o Default (%)",
        min_value=10.0,
        max_value=100.0,
        value=65.0,
        step=5.0,
    )
    funding_input = st.number_input(
        "Custo de Funding / CDI (% a.a.)",
        min_value=2.0,
        max_value=25.0,
        value=11.5,
        step=0.5,
    )

    st.markdown("---")
    st.caption(
        "Desenvolvido por **Renan Nocelli**  \nAnalista de Dados & Modelagem Estatística"
    )


# Executa simulação com os parâmetros selecionados
current_sim = sim_engine.simulate(
    cutoff_score=cutoff_input,
    application_volume=vol_input,
    avg_ticket=ticket_input,
    interest_rate_monthly_pct=interest_monthly,
    lgd_pct=lgd_input,
    funding_cost_annual_pct=funding_input,
)


# ==============================================================================
# HEADER PRINCIPAL & BANNER EXECUTIVO
# ==============================================================================
st.title("💳 Modelagem de Risco de Crédito & Credit Scoring")
st.markdown(
    "**Pipeline Econométrico Completo:** Binning Monotônico via *Weight of Evidence* (WoE), Seleção de Atributos por "
    "*Information Value* (IV), Regressão Logística Calibrada na Escala FICO (300-850) e Simulador Tático de Perda Esperada ($PD \\times LGD \\times EAD$)."
)

# LINHA DE CARDS DE KPI (MÉTRICAS DO CORTE ATIVO)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Taxa de Aprovação</div>
            <div class="metric-value">{current_sim.approval_rate_pct:.1f}%</div>
            <div class="metric-sub">{current_sim.approved_count:,} proponentes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Bad Rate da Safra</div>
            <div class="metric-value" style="color: {"#10b981" if current_sim.bad_rate_approved_pct <= 5.0 else "#f59e0b"};">
                {current_sim.bad_rate_approved_pct:.2f}%
            </div>
            <div class="metric-sub">{current_sim.bads_approved_count:,} inadimplentes previstos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Capital Concedido</div>
            <div class="metric-value">R$ {current_sim.approved_capital_million:.2f}M</div>
            <div class="metric-sub">Tíquete R$ {ticket_input:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Perda Esperada (EL)</div>
            <div class="metric-value" style="color: #ef4444;">R$ {current_sim.expected_loss_million:.2f}M</div>
            <div class="metric-sub">LGD {lgd_input:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Margem Líquida</div>
            <div class="metric-value" style="color: #38bdf8;">R$ {current_sim.net_credit_margin_million:.2f}M</div>
            <div class="metric-sub">Spread de {current_sim.net_credit_margin_pct:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ==============================================================================
# NAVEGAÇÃO EM ABAS
# ==============================================================================
tab_iv, tab_scorecard, tab_metrics, tab_simulator, tab_methodology = st.tabs(
    [
        "📊 1. Matriz WoE & Information Value",
        "🧮 2. Scorecard FICO & Avaliação Individual",
        "📈 3. Discriminação (K-S & ROC/AUC)",
        "🎯 4. Simulador Tático What-If (P&L)",
        "📐 5. Metodologia & Pipeline Matemático",
    ]
)


# ==============================================================================
# ABA 1: MATRIZ WOE & INFORMATION VALUE
# ==============================================================================
with tab_iv:
    st.subheader("Matriz de Seleção de Atributos por Information Value (IV)")
    st.markdown(
        "O **Information Value (IV)** quantifica a capacidade de cada variável de separar clientes adimplentes (*Goods*) "
        "de inadimplentes (*Bads*). Variáveis com $IV > 0.10$ são selecionadas para compor o Scorecard."
    )

    transformer = scorecard_model.woe_transformer
    summary_iv = transformer.get_summary_table()

    # Gráfico de barras horizontais do IV
    fig_iv = px.bar(
        summary_iv,
        x="Information Value (IV)",
        y="Variável",
        orientation="h",
        color="Poder Preditivo",
        color_discrete_map={
            "Forte": "#10b981",
            "Médio": "#38bdf8",
            "Fraco": "#f59e0b",
            "Inútil / Desprezível": "#ef4444",
        },
        title="Ranking de Information Value (IV) das Características",
        text="Information Value (IV)",
    )
    fig_iv.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"categoryorder": "total ascending"},
        height=350,
    )
    fig_iv.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig_iv, use_container_width=True)

    st.markdown("---")
    st.subheader("Inspeção Monotônica por Variável")

    selected_feat = st.selectbox(
        "Selecione uma variável para inspecionar os bins e a monotonicidade do WoE:",
        options=list(transformer.features_iv.keys()),
        format_func=lambda x: (
            f"{transformer.features_iv[x].feature_name} (IV: {transformer.features_iv[x].iv:.3f})"
        ),
    )

    feat_data = transformer.features_iv[selected_feat]
    col_g1, col_g2 = st.columns([1, 1])

    bin_labels = [b.bin_label for b in feat_data.bins]
    goods_pct = [b.dist_goods for b in feat_data.bins]
    bads_pct = [b.dist_bads for b in feat_data.bins]
    woe_vals = [b.woe for b in feat_data.bins]
    bad_rates = [b.bad_rate for b in feat_data.bins]

    with col_g1:
        # Gráfico Composto: Distribuição Goods vs Bads + Linha WoE
        fig_bins = go.Figure()
        fig_bins.add_trace(
            go.Bar(
                name="% Dist Goods", x=bin_labels, y=goods_pct, marker_color="#10b981"
            )
        )
        fig_bins.add_trace(
            go.Bar(name="% Dist Bads", x=bin_labels, y=bads_pct, marker_color="#ef4444")
        )
        fig_bins.add_trace(
            go.Scatter(
                name="Weight of Evidence (WoE)",
                x=bin_labels,
                y=woe_vals,
                mode="lines+markers+text",
                text=[f"{w:.2f}" for w in woe_vals],
                textposition="top center",
                yaxis="y2",
                line={"color": "#38bdf8", "width": 3},
            )
        )

        fig_bins.update_layout(
            title=f"Distribuição & Curva WoE: {feat_data.feature_name}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis={"title": "% da População"},
            yaxis2={"title": "WoE", "overlaying": "y", "side": "right"},
            barmode="group",
            height=380,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        st.plotly_chart(fig_bins, use_container_width=True)

    with col_g2:
        # Gráfico de Taxa de Inadimplência Real por Faixa
        fig_br = px.bar(
            x=bin_labels,
            y=bad_rates,
            text=[f"{br:.1f}%" for br in bad_rates],
            title=f"Bad Rate Empírica por Faixa: {feat_data.feature_name}",
            labels={"x": "Faixa / Bin", "y": "Bad Rate (%)"},
            color=bad_rates,
            color_continuous_scale="Reds",
        )
        fig_br.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            height=380,
        )
        fig_br.update_traces(textposition="outside")
        st.plotly_chart(fig_br, use_container_width=True)

    # Tabela detalhada de contingência
    st.write("#### Tabela de Contingência & Contribuição para o Information Value")
    rows = []
    for b in feat_data.bins:
        rows.append(
            {
                "Faixa (Bin)": b.bin_label,
                "Volume Total": f"{b.total:,}",
                "Bons Pagadores": f"{b.goods:,}",
                "Maus Pagadores": f"{b.bads:,}",
                "Bad Rate (%)": f"{b.bad_rate:.2f}%",
                "Dist. Goods (%)": f"{b.dist_goods:.2f}%",
                "Dist. Bads (%)": f"{b.dist_bads:.2f}%",
                "WoE": f"{b.woe:.4f}",
                "Contribuição IV": f"{b.iv_contribution:.4f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==============================================================================
# ABA 2: SCORECARD FICO & AVALIAÇÃO INDIVIDUAL
# ==============================================================================
with tab_scorecard:
    st.subheader("Scorecard FICO Calibrado & Simulador de Proponente em Tempo Real")
    st.markdown(
        "A escala de pontuação foi calibrada pelo método econométrico FICO clássico:  \n"
        "$$\\text{Factor} = \\frac{\\text{PDO}}{\\ln(2)} = \\frac{20}{\\ln(2)} \\approx 28.85, \\quad "
        "\\text{Offset} = \\text{Target Score} - \\text{Factor} \\cdot \\ln(\\text{Target Odds})$$  \n"
        "Com Score base de **600 pontos para Odds de 50:1** e **PDO de 20 pontos** (dobra a chance de adimplência a cada 20 pontos de score)."
    )

    col_sim_input, col_sim_output = st.columns([1.1, 0.9])

    with col_sim_input:
        st.markdown("#### 📝 Simule um Novo Proponente")
        with st.form("form_single_applicant"):
            dti_in = st.slider("Comprometimento de Renda (DTI %)", 5.0, 75.0, 22.0, 1.0)
            delinq_in = st.selectbox(
                "Atrasos 30-59 Dias (Últimos 24 meses)", [0, 1, 2, 3, 4], index=0
            )
            revolving_in = st.slider(
                "Utilização do Limite Rotativo (%)", 2.0, 98.0, 28.0, 1.0
            )
            tenure_in = st.slider("Tempo de Emprego Atual (Anos)", 0.2, 20.0, 4.5, 0.5)
            bureau_in = st.selectbox(
                "Consultas ao Bureau nos últimos 6 meses", [0, 1, 2, 3, 4, 6], index=1
            )
            age_in = st.slider("Idade do Tomador", 18, 75, 36, 1)

            submit_eval = st.form_submit_button(
                "⚡ Calcular Score FICO & Parecer", use_container_width=True
            )

        # Monta proponente
        applicant_df = pd.DataFrame(
            [
                {
                    "applicant_id": "PROP-NOVO",
                    "age": age_in,
                    "job_tenure": tenure_in,
                    "income": 6500.0,
                    "requested_amount": ticket_input,
                    "dti": dti_in,
                    "revolving_util": revolving_in,
                    "delinquency_2y": delinq_in,
                    "bureau_inquiries": bureau_in,
                }
            ]
        )

        ind_score = int(scorecard_model.predict_score(applicant_df).iloc[0])
        ind_pd = (
            float(scorecard_model.predict_proba_default(applicant_df).iloc[0]) * 100.0
        )
        is_approved = ind_score >= cutoff_input

    with col_sim_output:
        st.markdown("#### 🎯 Parecer da Esteira de Concessão")

        # Gauge Chart com Score FICO
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=ind_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Score FICO Calculado", "font": {"size": 22}},
                delta={
                    "reference": cutoff_input,
                    "increasing": {"color": "#10b981"},
                    "decreasing": {"color": "#ef4444"},
                },
                gauge={
                    "axis": {"range": [300, 850], "tickwidth": 1, "tickcolor": "white"},
                    "bar": {"color": "#38bdf8", "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 1,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [300, 480], "color": "rgba(239, 68, 68, 0.4)"},
                        {"range": [480, 580], "color": "rgba(245, 158, 11, 0.4)"},
                        {"range": [580, 700], "color": "rgba(56, 189, 248, 0.4)"},
                        {"range": [700, 850], "color": "rgba(16, 185, 129, 0.4)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 4},
                        "thickness": 0.8,
                        "value": cutoff_input,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        status_class = "badge-approved" if is_approved else "badge-rejected"
        status_text = (
            "APROVADO PELA POLÍTICA"
            if is_approved
            else "REPROVADO POR POLÍTICA DE RISCO"
        )
        status_color = "#10b981" if is_approved else "#ef4444"

        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 5px solid {status_color};">
                <div style="font-size: 1.2rem; font-weight: 700; color: {status_color}; margin-bottom: 8px;">
                    {status_text}
                </div>
                <div style="font-size: 0.95rem; color: #cbd5e1;">
                    • <b>Score FICO:</b> {ind_score} pontos (Nota de Corte: {cutoff_input})<br/>
                    • <b>Probabilidade de Default (PD):</b> {ind_pd:.2f}%<br/>
                    • <b>Risco Relativo:</b> {"Baixo Risco (Prime)" if ind_score >= 680 else ("Risco Moderado" if ind_score >= 540 else "Alto Risco (Subprime)")}<br/>
                    • <b>LGD Estimada:</b> {lgd_input:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Tabela de Regras e Pontuação do Scorecard FICO")
    rules_df = scorecard_model.get_scorecard_dataframe()
    st.dataframe(rules_df, use_container_width=True, hide_index=True)


# ==============================================================================
# ABA 3: VALIDAÇÃO DISCRIMINATÓRIA (K-S & ROC/AUC)
# ==============================================================================
with tab_metrics:
    st.subheader("Avaliação Estatística de Discriminação & Qualidade do Modelo")
    st.markdown(
        "A qualidade de um scorecard de crédito é validada principalmente por duas métricas consagradas pelo comitê de Basileia:  \n"
        "1. **Estatística Kolmogorov-Smirnov ($K\\text{-}S$):** Mede a separação máxima entre a função de distribuição acumulada "
        "dos bons pagadores e dos maus pagadores ($K\\text{-}S = \\max |F_{Bads}(s) - F_{Goods}(s)|$).  \n"
        "2. **Curva ROC e Coeficiente Gini:** A área sob a curva ROC ($AUC$) e a transformação linear $Gini = 2 \\cdot AUC - 1$."
    )

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric(
            "Estatística K-S",
            f"{perf_metrics.ks_stat:.1f}%",
            f"Corte de Máx. Separação: {perf_metrics.max_ks_score} pts",
        )
    with c_m2:
        st.metric(
            "Coeficiente Gini", f"{perf_metrics.gini:.1f}%", "Padrão de Mercado > 45%"
        )
    with c_m3:
        st.metric(
            "Área sob a Curva (ROC-AUC)",
            f"{perf_metrics.auc:.3f}",
            "Excelente Discriminação",
        )
    with c_m4:
        st.metric(
            "Bad Rate Global (População)",
            f"{df_raw['default'].mean() * 100:.1f}%",
            "45.000 proponentes",
        )

    col_ks, col_roc = st.columns(2)

    with col_ks:
        # Gráfico K-S (CDFs Acumuladas)
        ks_table = perf_metrics.ks_curve_data
        fig_ks = go.Figure()

        # Eixo x como o score médio da faixa
        x_scores = [
            round(val) if not np.isnan(val) else 500 for val in ks_table["mean_score"]
        ]

        fig_ks.add_trace(
            go.Scatter(
                x=x_scores,
                y=ks_table["cum_bads_pct"],
                mode="lines+markers",
                name="CDF Inadimplentes (Bads)",
                line={"color": "#ef4444", "width": 3},
            )
        )
        fig_ks.add_trace(
            go.Scatter(
                x=x_scores,
                y=ks_table["cum_goods_pct"],
                mode="lines+markers",
                name="CDF Bons Pagadores (Goods)",
                line={"color": "#10b981", "width": 3},
            )
        )

        # Linha vertical do K-S Máximo
        fig_ks.add_vline(
            x=perf_metrics.max_ks_score,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"K-S = {perf_metrics.ks_stat:.1f}% ({perf_metrics.max_ks_score} pts)",
            annotation_position="top left",
        )

        fig_ks.update_layout(
            title="Curvas de Distribuição Acumulada & Kolmogorov-Smirnov (K-S)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Score de Crédito FICO",
            yaxis_title="% Acumulado da População",
            height=380,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        st.plotly_chart(fig_ks, use_container_width=True)

    with col_roc:
        # Curva ROC
        roc_data = perf_metrics.roc_curve_data
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                name=f"Scorecard FICO (AUC = {perf_metrics.auc:.3f})",
                line={"color": "#38bdf8", "width": 3},
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.15)",
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Baseline Aleatório (AUC = 0.500)",
                line={"color": "#64748b", "dash": "dash"},
            )
        )
        fig_roc.update_layout(
            title=f"Curva ROC & Discriminação (Gini = {perf_metrics.gini:.1f}%)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Taxa de Falso Positivo (FPR)",
            yaxis_title="Taxa de Verdadeiro Positivo (TPR / Recall)",
            height=380,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        st.plotly_chart(fig_roc, use_container_width=True)


# ==============================================================================
# ABA 4: SIMULADOR TÁTICO WHAT-IF (P&L & GESTÃO DE RISCO)
# ==============================================================================
with tab_simulator:
    st.subheader("Fronteira Eficiente de Concessão de Crédito & Gestão de P&L")
    st.markdown(
        "Ajuste e visualize a curva de sensibilidade entre volume aprovado, inadimplência esperada e lucratividade líquida. "
        "A nota de corte ideal maximiza o lucro da esteira equilibrando receita de juros contra a Perda Esperada ($EL$)."
    )

    # Gera fronteira para os parâmetros atuais
    frontier_df = sim_engine.generate_cutoff_frontier(
        min_cutoff=360,
        max_cutoff=740,
        step=20,
        application_volume=vol_input,
        avg_ticket=ticket_input,
    )

    # Gráfico Duplo de Fronteira: Aprovação vs Bad Rate e Lucro Líquido
    fig_front = go.Figure()
    fig_front.add_trace(
        go.Scatter(
            x=frontier_df["Cutoff Score"],
            y=frontier_df["Taxa Aprovação (%)"],
            mode="lines+markers",
            name="Taxa de Aprovação (%)",
            line={"color": "#10b981", "width": 2.5},
        )
    )
    fig_front.add_trace(
        go.Scatter(
            x=frontier_df["Cutoff Score"],
            y=frontier_df["Bad Rate (%)"],
            mode="lines+markers",
            name="Bad Rate Safra (%)",
            line={"color": "#ef4444", "width": 2.5},
        )
    )
    fig_front.add_trace(
        go.Scatter(
            x=frontier_df["Cutoff Score"],
            y=frontier_df["Margem Líquida (R$ M)"],
            mode="lines+markers",
            name="Margem Líquida (R$ M)",
            yaxis="y2",
            line={"color": "#38bdf8", "width": 3, "dash": "dot"},
        )
    )

    # Ponto de corte atualmente ativo
    fig_front.add_vline(
        x=cutoff_input,
        line_dash="dash",
        line_color="#ffffff",
        annotation_text=f"Corte Ativo: {cutoff_input} pts",
        annotation_position="top left",
    )

    fig_front.update_layout(
        title="Curva de Trade-off de Concessão: Volume vs Qualidade vs Margem",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Nota de Corte (Cut-off Score)",
        yaxis={"title": "Percentual (%)"},
        yaxis2={"title": "Margem Líquida (R$ M)", "overlaying": "y", "side": "right"},
        height=420,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    st.plotly_chart(fig_front, use_container_width=True)

    st.write("#### Tabela de Sensibilidade em Múltiplos Pedaços de Corte")
    st.dataframe(frontier_df, use_container_width=True, hide_index=True)


# ==============================================================================
# ABA 5: METODOLOGIA & PIPELINE MATEMÁTICO
# ==============================================================================
with tab_methodology:
    st.subheader("Fundamentação Teórica & Formulações Econométricas")

    st.markdown(
        """
        ### 1. Weight of Evidence (WoE) & Information Value (IV)
        A transformação **Weight of Evidence** quantifica a força de evidência de uma faixa $i$ de um atributo em favorecer o evento de adimplência em relação à inadimplência:

        $$\\text{WoE}_i = \\ln\\left( \\frac{\\% \\text{Goods}_i}{\\% \\text{Bads}_i} \\right) = \\ln\\left( \\frac{G_i / G_T}{B_i / B_T} \\right)$$

        O **Information Value (IV)** afere o poder discriminatório global da variável $k$:

        $$\\text{IV}_k = \\sum_{i=1}^{m} \\left( \\frac{G_i}{G_T} - \\frac{B_i}{B_T} \\right) \\cdot \\text{WoE}_i$$

        ---

        ### 2. Calibração do Scorecard na Escala FICO
        Dada a probabilidade de default $p$ estimada pela regressão logística nos atributos transformados por WoE:

        $$\\ln(\\text{Odds}) = \\beta_0 + \\sum_{j=1}^{p} \\beta_j \\cdot \\text{WoE}_j, \\quad \\text{onde } \\text{Odds} = \\frac{1-p}{p}$$

        A escala FICO é calibrada através dos parâmetros:
        - **Factor:** $\\text{Factor} = \\frac{\\text{PDO}}{\\ln(2)}$
        - **Offset:** $\\text{Offset} = \\text{Target Score} - \\text{Factor} \\cdot \\ln(\\text{Target Odds})$
        - **Score Final:** $\\text{Score} = \\text{Offset} + \\text{Factor} \\cdot \\ln(\\text{Odds})$

        ---

        ### 3. Gestão de Risco & Perda Esperada (Expected Loss)
        O provisionamento e impacto contábil de crédito segue as diretrizes de **Basileia II / IFRS 9**:

        $$\\text{EL} = \\text{PD} \\times \\text{LGD} \\times \\text{EAD}$$

        Onde:
        - $\\text{PD}$ (*Probability of Default*): Estimada pelo modelo estatístico.
        - $\\text{LGD}$ (*Loss Given Default*): Percentual de perda efetiva após esforços de recuperação (ex: 65%).
        - $\\text{EAD}$ (*Exposure at Default*): Saldo devedor exposto no momento do inadimplemento.
        """
    )

st.markdown("---")
st.caption(
    "© 2026 Renan Nocelli. Projeto Integrante do Portfólio de Ciência de Dados e Modelagem Estatística: "
    "[renan-nocelli.vercel.app/projetos/credit-scoring-risco-credito](https://renan-nocelli.vercel.app/projetos/credit-scoring-risco-credito)"
)
