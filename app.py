"""
app.py
======
Dashboard Executivo de Modelagem Estatística de Risco de Crédito & Credit Scoring.
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
# CONFIGURAÇÃO DA PÁGINA & DESIGN SYSTEM (MODERN MINIMALIST FINTECH THEME)
# ==============================================================================
st.set_page_config(
    page_title="Credit Scoring & Risco de Crédito | Renan Nocelli",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta Moderna Desaturada (Estilo Linear / Stripe / Bloomberg Moderno)
COLOR_GOOD = "#34d399"  # Muted Sage / Verde Suave
COLOR_BAD = "#fb7185"  # Muted Rose / Coral Suave
COLOR_PRIMARY = "#818cf8"  # Soft Indigo / Lavanda Profissional
COLOR_SECONDARY = "#38bdf8"  # Slate Blue / Ciano Desaturado
COLOR_WARNING = "#fbbf24"  # Muted Amber / Dourado Suave
COLOR_NEUTRAL = "#94a3b8"  # Slate Grey

CUSTOM_CSS = """
<style>
    /* Estilização Geral e Tipografia */
    .main {
        background-color: #0b0f17;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Cartões de Métricas Executivas */
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 16px 18px;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #334155;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 600;
        color: #f1f5f9;
        line-height: 1.15;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Badges de Decisão */
    .status-badge-container {
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
    }
    .badge-approved-container {
        background-color: rgba(52, 211, 153, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.2);
    }
    .badge-rejected-container {
        background-color: rgba(251, 113, 133, 0.08);
        border: 1px solid rgba(251, 113, 133, 0.2);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# CARREGAMENTO EM CACHE DOS DADOS E MODELO
# ==============================================================================
@st.cache_resource(show_spinner="Carregando e treinando pipeline de Scorecard FICO...")
def load_trained_system():
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


# Função utilitária para aplicar layout moderno e sóbrio aos gráficos Plotly
def apply_modern_layout(fig, title: str = "", height: int = 380):
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 14, "color": "#e2e8f0", "family": "sans-serif"},
        },
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        font={"family": "sans-serif", "color": "#94a3b8", "size": 11},
        xaxis={
            "gridcolor": "rgba(255, 255, 255, 0.05)",
            "zerolinecolor": "rgba(255, 255, 255, 0.08)",
            "tickfont": {"color": "#94a3b8"},
        },
        yaxis={
            "gridcolor": "rgba(255, 255, 255, 0.05)",
            "zerolinecolor": "rgba(255, 255, 255, 0.08)",
            "tickfont": {"color": "#94a3b8"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11, "color": "#cbd5e1"},
        },
    )
    return fig


# ==============================================================================
# BARRA LATERAL (CONTROLES DO SIMULADOR TÁTICO)
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 6px 0 16px 0;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600;">
                Sistema de Decisão
            </div>
            <div style="font-size: 1.15rem; font-weight: 600; color: #f8fafc; margin-top: 2px;">
                Parâmetros da Esteira
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cutoff_input = st.slider(
        "Ponto de Corte (Cut-off Score)",
        min_value=320,
        max_value=760,
        value=540,
        step=10,
        help="Proponentes com pontuação igual ou superior a este limiar serão aprovados na esteira.",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 12px;'>Premissas da Carteira</div>",
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
            <b>Renan Nocelli</b><br>
            Modelagem Estatística & Risco de Crédito<br>
            Portfolio: renan-nocelli.vercel.app
        </div>
        """,
        unsafe_allow_html=True,
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
st.markdown(
    """
    <div style="margin-bottom: 24px;">
        <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #818cf8; margin-bottom: 4px;">
            Engenharia de Risco Financeiro & Estatística Aplicada
        </div>
        <div style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">
            Modelagem de Risco de Crédito & Credit Scoring
        </div>
        <div style="font-size: 0.92rem; color: #94a3b8; margin-top: 6px; max-width: 900px; line-height: 1.5;">
            Pipeline estatístico com transformação monotônica por Weight of Evidence (WoE), seleção por 
            Information Value (IV), calibração econométrica FICO (300-850) e simulação de Perda Esperada (PD × LGD × EAD).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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
    color_br = COLOR_GOOD if current_sim.bad_rate_approved_pct <= 5.0 else COLOR_WARNING
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Bad Rate da Safra</div>
            <div class="metric-value" style="color: {color_br};">
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
            <div class="metric-sub">Tíquete médio R$ {ticket_input:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Perda Esperada (EL)</div>
            <div class="metric-value" style="color: {COLOR_BAD};">R$ {current_sim.expected_loss_million:.2f}M</div>
            <div class="metric-sub">LGD calibrada em {lgd_input:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Margem Líquida</div>
            <div class="metric-value" style="color: {COLOR_PRIMARY};">R$ {current_sim.net_credit_margin_million:.2f}M</div>
            <div class="metric-sub">Spread líquido de {current_sim.net_credit_margin_pct:.1f}%</div>
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
        "Matriz WoE & Information Value",
        "Scorecard FICO & Avaliação Individual",
        "Discriminação (K-S & ROC/AUC)",
        "Simulador Tático What-If (P&L)",
        "Metodologia & Pipeline Matemático",
    ]
)


# ==============================================================================
# ABA 1: MATRIZ WOE & INFORMATION VALUE
# ==============================================================================
with tab_iv:
    st.subheader("Matriz de Seleção de Atributos por Information Value (IV)")
    st.markdown(
        "O Information Value (IV) avalia o poder discriminatório de cada característica na esteira de risco. "
        "Atributos com IV superior a 0.10 são considerados elegíveis para a arquitetura do Scorecard."
    )

    transformer = scorecard_model.woe_transformer
    summary_iv = transformer.get_summary_table()

    # Gráfico de barras horizontais do IV com cores sóbrias
    fig_iv = px.bar(
        summary_iv,
        x="Information Value (IV)",
        y="Variável",
        orientation="h",
        color="Poder Preditivo",
        color_discrete_map={
            "Forte": COLOR_GOOD,
            "Médio": COLOR_SECONDARY,
            "Fraco": COLOR_WARNING,
            "Inútil / Desprezível": COLOR_BAD,
        },
        text="Information Value (IV)",
    )
    apply_modern_layout(
        fig_iv, title="Ranking de Information Value (IV) por Variável", height=320
    )
    fig_iv.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_iv.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig_iv, use_container_width=True)

    st.markdown("---")
    st.subheader("Análise Monotônica por Variável")

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
                name="% Dist Goods",
                x=bin_labels,
                y=goods_pct,
                marker_color=COLOR_GOOD,
                opacity=0.85,
            )
        )
        fig_bins.add_trace(
            go.Bar(
                name="% Dist Bads",
                x=bin_labels,
                y=bads_pct,
                marker_color=COLOR_BAD,
                opacity=0.85,
            )
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
                line={"color": COLOR_PRIMARY, "width": 2.5},
            )
        )

        apply_modern_layout(
            fig_bins,
            title=f"Distribuição & Curva WoE: {feat_data.feature_name}",
            height=360,
        )
        fig_bins.update_layout(
            yaxis={"title": "% da População", "gridcolor": "rgba(255,255,255,0.05)"},
            yaxis2={
                "title": "WoE",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "tickfont": {"color": COLOR_PRIMARY},
            },
            barmode="group",
        )
        st.plotly_chart(fig_bins, use_container_width=True)

    with col_g2:
        # Gráfico de Taxa de Inadimplência Real por Faixa
        fig_br = px.bar(
            x=bin_labels,
            y=bad_rates,
            text=[f"{br:.1f}%" for br in bad_rates],
            labels={"x": "Faixa / Bin", "y": "Bad Rate (%)"},
            color_discrete_sequence=[COLOR_SECONDARY],
        )
        apply_modern_layout(
            fig_br,
            title=f"Bad Rate Empírica por Faixa: {feat_data.feature_name}",
            height=360,
        )
        fig_br.update_traces(textposition="outside", marker={"opacity": 0.85})
        st.plotly_chart(fig_br, use_container_width=True)

    # Tabela detalhada de contingência
    st.markdown("#### Tabela de Contingência & Contribuição para o Information Value")
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
    st.subheader("Scorecard FICO Calibrado & Avaliação Individual de Proponente")
    st.markdown(
        "A régua de pontuação utiliza a formulação econométrica clássica FICO (300 a 850 pontos):  \n"
        "$$\\text{Factor} = \\frac{\\text{PDO}}{\\ln(2)} \\approx 28.85, \\quad "
        "\\text{Offset} = \\text{Target Score} - \\text{Factor} \\cdot \\ln(\\text{Target Odds}) \\approx 487.13$$  \n"
        "Calibrada com base de 600 pontos para odds de 50:1 e PDO de 20 pontos."
    )

    col_sim_input, col_sim_output = st.columns([1.1, 0.9])

    with col_sim_input:
        st.markdown("#### Simulação de Proponente")
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
                "Calcular Score FICO & Parecer", use_container_width=True
            )

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
        st.markdown("#### Parecer da Esteira")

        # Gauge Chart Minimalista e Sóbrio
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=ind_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={
                    "text": "Score FICO Calculado",
                    "font": {"size": 16, "color": "#cbd5e1"},
                },
                gauge={
                    "axis": {
                        "range": [300, 850],
                        "tickwidth": 1,
                        "tickcolor": "#64748b",
                    },
                    "bar": {"color": COLOR_PRIMARY, "thickness": 0.25},
                    "bgcolor": "rgba(255, 255, 255, 0.02)",
                    "borderwidth": 1,
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [300, 480], "color": "rgba(251, 113, 133, 0.15)"},
                        {"range": [480, 580], "color": "rgba(251, 191, 36, 0.15)"},
                        {"range": [580, 700], "color": "rgba(56, 189, 248, 0.15)"},
                        {"range": [700, 850], "color": "rgba(52, 211, 153, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f8fafc", "width": 3},
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
            height=260,
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
            font={"color": "#f8fafc", "family": "sans-serif"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        container_class = (
            "badge-approved-container" if is_approved else "badge-rejected-container"
        )
        status_text = (
            "APROVADO PELA POLÍTICA DE CRÉDITO"
            if is_approved
            else "REPROVADO POR POLÍTICA DE RISCO"
        )
        status_color = COLOR_GOOD if is_approved else COLOR_BAD

        st.markdown(
            f"""
            <div class="status-badge-container {container_class}">
                <div style="font-size: 1.05rem; font-weight: 600; color: {status_color}; margin-bottom: 8px;">
                    {status_text}
                </div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6;">
                    • <b>Score Individual:</b> {ind_score} pontos (Nota de Corte: {cutoff_input})<br/>
                    • <b>Probabilidade de Default (PD):</b> {ind_pd:.2f}%<br/>
                    • <b>Rating de Risco:</b> {"Prime (Baixo Risco)" if ind_score >= 680 else ("Moderado" if ind_score >= 540 else "Subprime (Alto Risco)")}<br/>
                    • <b>LGD Estimada:</b> {lgd_input:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Dicionário de Regras de Pontuação do Scorecard FICO")
    rules_df = scorecard_model.get_scorecard_dataframe()
    st.dataframe(rules_df, use_container_width=True, hide_index=True)


# ==============================================================================
# ABA 3: VALIDAÇÃO DISCRIMINATÓRIA (K-S & ROC/AUC)
# ==============================================================================
with tab_metrics:
    st.subheader("Validação Estatística de Discriminação")
    st.markdown(
        "A capacidade de separação do modelo é mensurada pela estatística de Kolmogorov-Smirnov (K-S) "
        "e pela curva Receiver Operating Characteristic (ROC), em conformidade com as recomendações de Basileia II."
    )

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric(
            "Estatística K-S",
            f"{perf_metrics.ks_stat:.1f}%",
            f"Corte de Máxima Separação: {perf_metrics.max_ks_score} pts",
        )
    with c_m2:
        st.metric(
            "Coeficiente Gini",
            f"{perf_metrics.gini:.1f}%",
            "Referência de Mercado > 45%",
        )
    with c_m3:
        st.metric(
            "Área sob a Curva (ROC-AUC)",
            f"{perf_metrics.auc:.3f}",
            "Forte Discriminação",
        )
    with c_m4:
        st.metric(
            "Bad Rate Global da Amostra",
            f"{df_raw['default'].mean() * 100:.1f}%",
            "45.000 proponentes",
        )

    col_ks, col_roc = st.columns(2)

    with col_ks:
        # Gráfico K-S (CDFs Acumuladas)
        ks_table = perf_metrics.ks_curve_data
        fig_ks = go.Figure()

        x_scores = [
            round(val) if not np.isnan(val) else 500 for val in ks_table["mean_score"]
        ]

        fig_ks.add_trace(
            go.Scatter(
                x=x_scores,
                y=ks_table["cum_bads_pct"],
                mode="lines+markers",
                name="CDF Inadimplentes (Bads)",
                line={"color": COLOR_BAD, "width": 2.5},
            )
        )
        fig_ks.add_trace(
            go.Scatter(
                x=x_scores,
                y=ks_table["cum_goods_pct"],
                mode="lines+markers",
                name="CDF Bons Pagadores (Goods)",
                line={"color": COLOR_GOOD, "width": 2.5},
            )
        )

        fig_ks.add_vline(
            x=perf_metrics.max_ks_score,
            line_dash="dash",
            line_color=COLOR_WARNING,
            annotation_text=f"K-S = {perf_metrics.ks_stat:.1f}%",
            annotation_position="top left",
            annotation_font={"color": COLOR_WARNING, "size": 11},
        )

        apply_modern_layout(
            fig_ks,
            title="Curvas de Distribuição Acumulada & Kolmogorov-Smirnov (K-S)",
            height=380,
        )
        fig_ks.update_layout(
            xaxis_title="Score de Crédito FICO",
            yaxis_title="% Acumulado da População",
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
                line={"color": COLOR_SECONDARY, "width": 2.5},
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.08)",
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Baseline Aleatório (AUC = 0.500)",
                line={"color": "#475569", "dash": "dash", "width": 1.5},
            )
        )
        apply_modern_layout(
            fig_roc,
            title=f"Curva ROC & Discriminação (Gini = {perf_metrics.gini:.1f}%)",
            height=380,
        )
        fig_roc.update_layout(
            xaxis_title="Taxa de Falso Positivo (FPR)",
            yaxis_title="Taxa de Verdadeiro Positivo (TPR / Recall)",
        )
        st.plotly_chart(fig_roc, use_container_width=True)


# ==============================================================================
# ABA 4: SIMULADOR TÁTICO WHAT-IF (P&L & GESTÃO DE RISCO)
# ==============================================================================
with tab_simulator:
    st.subheader("Fronteira de Concessão de Crédito & Gestão de P&L")
    st.markdown(
        "Avaliação tática da sensibilidade entre volume aprovado, inadimplência da carteira e margem financeira líquida. "
        "A definição do ponto de corte equilibra a geração de receita de juros contra a Perda Esperada (EL)."
    )

    frontier_df = sim_engine.generate_cutoff_frontier(
        min_cutoff=360,
        max_cutoff=740,
        step=20,
        application_volume=vol_input,
        avg_ticket=ticket_input,
    )

    fig_front = go.Figure()
    fig_front.add_trace(
        go.Scatter(
            x=frontier_df["Cutoff Score"],
            y=frontier_df["Taxa Aprovação (%)"],
            mode="lines+markers",
            name="Taxa de Aprovação (%)",
            line={"color": COLOR_GOOD, "width": 2},
        )
    )
    fig_front.add_trace(
        go.Scatter(
            x=frontier_df["Cutoff Score"],
            y=frontier_df["Bad Rate (%)"],
            mode="lines+markers",
            name="Bad Rate Safra (%)",
            line={"color": COLOR_BAD, "width": 2},
        )
    )
    fig_front.add_trace(
        go.Scatter(
            x=frontier_df["Cutoff Score"],
            y=frontier_df["Margem Líquida (R$ M)"],
            mode="lines+markers",
            name="Margem Líquida (R$ M)",
            yaxis="y2",
            line={"color": COLOR_PRIMARY, "width": 2.5, "dash": "dot"},
        )
    )

    fig_front.add_vline(
        x=cutoff_input,
        line_dash="dash",
        line_color="#f8fafc",
        annotation_text=f"Corte Ativo: {cutoff_input} pts",
        annotation_position="top left",
        annotation_font={"color": "#f8fafc", "size": 11},
    )

    apply_modern_layout(
        fig_front,
        title="Trade-off de Concessão: Volume vs Qualidade vs Margem Líquida",
        height=420,
    )
    fig_front.update_layout(
        xaxis_title="Nota de Corte (Cut-off Score)",
        yaxis={"title": "Percentual (%)", "gridcolor": "rgba(255,255,255,0.05)"},
        yaxis2={
            "title": "Margem Líquida (R$ M)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "tickfont": {"color": COLOR_PRIMARY},
        },
    )
    st.plotly_chart(fig_front, use_container_width=True)

    st.markdown("#### Tabela de Sensibilidade em Múltiplos Pontos de Corte")
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

        A calibração na escala FICO é estruturada através dos parâmetros:
        - **Factor:** $\\text{Factor} = \\frac{\\text{PDO}}{\\ln(2)}$
        - **Offset:** $\\text{Offset} = \\text{Target Score} - \\text{Factor} \\cdot \\ln(\\text{Target Odds})$
        - **Score Final:** $\\text{Score} = \\text{Offset} + \\text{Factor} \\cdot \\ln(\\text{Odds})$

        ---

        ### 3. Gestão de Risco & Perda Esperada (Expected Loss)
        O provisionamento e apuração contábil seguem os preceitos de **Basileia II / IFRS 9**:

        $$\\text{EL} = \\text{PD} \\times \\text{LGD} \\times \\text{EAD}$$

        Onde:
        - $\\text{PD}$ (*Probability of Default*): Estimada pelo modelo estatístico.
        - $\\text{LGD}$ (*Loss Given Default*): Percentual de perda efetiva após esforços de recuperação judicial e amigável.
        - $\\text{EAD}$ (*Exposure at Default*): Saldo devedor exposto no momento do inadimplemento.
        """
    )

st.markdown("---")
st.caption(
    "Renan Nocelli. Projeto integrante do Portfólio de Ciência de Dados e Modelagem Estatística: "
    "[renan-nocelli.vercel.app/projetos/credit-scoring-risco-credito](https://renan-nocelli.vercel.app/projetos/credit-scoring-risco-credito)"
)
