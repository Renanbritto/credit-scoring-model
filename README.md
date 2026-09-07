# 💳 Statistical Credit Scoring & Scorecard FICO (WoE, IV, KS & Gini)

[![CI Quality & Test Pipeline](https://github.com/Renanbritto/credit-scoring-model/actions/workflows/ci.yml/badge.svg)](https://github.com/Renanbritto/credit-scoring-model/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Testing: Pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://pytest.org/)
[![Framework: Streamlit](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![Portfolio](https://img.shields.io/badge/Portfolio-renan--nocelli.vercel.app-7928CA.svg)](https://renan-nocelli.vercel.app/projetos/credit-scoring-risco-credito)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pipeline de Engenharia e Ciência de Dados para modelagem econométrica de **Risco de Crédito**, transformação monotônica por **Weight of Evidence (WoE)**, seleção de atributos por **Information Value (IV)**, calibração na escala **FICO (300-850)**, validação discriminatória com **Kolmogorov-Smirnov ($K\text{-}S$)**, **Curva ROC / AUC** e **simulador executivo What-If de ponto de corte (Cut-off) e Perda Esperada ($PD \times LGD \times EAD$)** para esteira de concessão de crédito.

---

## 🎯 Desafio de Negócio & Motivação

Esteiras tradicionais de concessão de crédito enfrentam dois gargalos estruturais:
1. **Falso Positivo (Perda de Oportunidade):** Reprovar proponentes solventes ao adotar políticas de corte arbitrárias sem calibração estatística de odds, desperdiçando receita com spread líquido.
2. **Falso Negativo (Inadimplência Oculta):** Aprovar proponentes com sobre-endividamento camuflado ou comportamento de risco recente, gerando inadimplência severa e perdas contábeis diretas.
3. **Falta de Previsibilidade de Perda Esperada ($EL$):** Dificuldade de mensurar em tempo real o impacto de mudanças na nota de corte (Cut-off) sobre a taxa de aprovação, inadimplência da safra (*Bad Rate*), consumo de capital e margem financeira líquida da instituição.

Este projeto resolve essas dores através de um **pipeline de modelagem estatística auditável**, fundamentado nas melhores práticas de **Basileia II / IFRS 9** e escalonamento na clássica régua **FICO**.

---

## 📐 Fundamentação Estatística & Econométrica

### 1. Weight of Evidence (WoE)
A transformação **Weight of Evidence** lineariza a relação entre variáveis contínuas/discretas e as *log-odds* da inadimplência, garantindo monotonicidade e estabilidade contra outliers:

$$\text{WoE}_i = \ln\left( \frac{\% \text{Goods}_i}{\% \text{Bads}_i} \right) = \ln\left( \frac{G_i / G_T}{B_i / B_T} \right)$$

Onde:
- $G_i, B_i$: Contagem de bons (*Goods*) e maus pagadores (*Bads*) na faixa $i$.
- $G_T, B_T$: Total acumulado de bons e maus na amostra de calibração.

Valores positivos de $\text{WoE}$ indicam que a faixa possui maior densidade de bons pagadores em relação à média geral.

---

### 2. Information Value (IV) & Seleção de Features
O **Information Value** mede o poder discriminatório global de uma variável independente, orientando a seleção parcimoniosa de atributos:

$$\text{IV}_k = \sum_{i=1}^{m} \left( \frac{G_i}{G_T} - \frac{B_i}{B_T} \right) \cdot \text{WoE}_i$$

**Critério de Siddiqi (Credit Risk Scorecards):**
| Information Value (IV) | Poder Preditivo | Ação na Modelagem |
|---|---|---|
| $< 0.02$ | Inútil / Desprezível | Descartar |
| $0.02 - 0.10$ | Fraco | Descartar / Usar apenas com forte justificativa de negócio |
| $0.10 - 0.30$ | Médio | **Selecionar** para o modelo |
| $0.30 - 0.50$ | Forte | **Selecionar** (variáveis âncoras da decisão) |
| $> 0.50$ | Muito Forte | Investigar risco de *data leakage* / superexposição |

---

### 3. Regressão Logística & Calibração FICO
A probabilidade de inadimplência $p = P(\text{Default} = 1)$ é modelada sobre as variáveis transformadas em $\text{WoE}$:

$$\ln\left( \frac{p}{1 - p} \right) = \beta_0 + \sum_{j=1}^{p} \beta_j \cdot \text{WoE}_j$$

A calibração na escala de pontos **FICO (300 - 850)** segue as especificações de mercado:
- **Base Score:** 600 pontos para $\text{Odds} = 50:1$ ($\text{Target Odds} = \frac{\text{Goods}}{\text{Bads}} = 50$).
- **PDO (Points to Double the Odds):** 20 pontos (a cada 20 pontos de aumento no score, as odds de adimplência dobram).

$$\text{Factor} = \frac{\text{PDO}}{\ln(2)} = \frac{20}{\ln(2)} \approx 28.8539$$

$$\text{Offset} = \text{Target Score} - \text{Factor} \cdot \ln(\text{Target Odds}) = 600 - 28.8539 \cdot \ln(50) \approx 487.13$$

$$\text{Score} = \text{Offset} + \text{Factor} \cdot \ln\left(\frac{1 - p}{p}\right)$$

---

### 4. Validação Discriminatória: Kolmogorov-Smirnov ($K\text{-}S$) & Gini
- **Estatística $K\text{-}S$:** Quantifica a distância máxima vertical entre as funções de distribuição acumulada ($\text{CDF}$) dos maus e bons pagadores:
  $$K\text{-}S = \max_{s} | F_{\text{Bads}}(s) - F_{\text{Goods}}(s) |$$
  *(Benchmark de mercado: $K\text{-}S > 40\%$ representa excelente capacidade discriminatória).*

- **Curva ROC, AUC e Coeficiente Gini:**
  $$\text{Gini} = 2 \cdot \text{AUC} - 1$$

---

### 5. Gestão de Risco, Perda Esperada e Margem de Spread
Para um ponto de corte ($\text{Cut-off}$) definido:
1. **Perda Esperada (Expected Loss):**
   $$\text{EL} = \text{EAD} \times \text{PD} \times \text{LGD}$$
2. **Margem Financeira Líquida da Carteira:**
   $$\text{Margem Líquida} = \text{Receita Bruta de Juros} - \text{Custo de Funding (CDI)} - \text{Perda Esperada (EL)}$$

---

## 🏗️ Arquitetura do Repositório

```
credit-scoring-model/
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline de Integração Contínua (Ruff + Pytest)
├── .gitignore                   # Arquivos ignorados pelo Git
├── README.md                    # Documentação técnica e de negócio completa
├── requirements.txt             # Dependências do projeto
├── app.py                       # Aplicação Web Interativa em Streamlit
├── data/
│   ├── raw/
│   │   └── credit_applications_sample.csv   # Amostra dos 45.000 proponentes
│   └── processed/
│       └── scorecard_dictionary.csv         # Dicionário calibrado de regras FICO
├── scripts/
│   └── build_data.py            # Script para regenerar base empírica e scorecard
├── src/
│   ├── __init__.py
│   ├── data_generator.py        # Gerador sintético com correlações de crédito
│   ├── woe_iv.py                # Engine de binning, WoE e Information Value
│   ├── scorecard.py             # Regressão logística, scaling FICO e pontos por bin
│   ├── metrics.py               # Cálculos de KS, ROC/AUC, Gini e deciles
│   └── simulator.py             # Simulador de Cut-off, Perda Esperada e P&L
└── tests/
    ├── __init__.py
    ├── test_woe_iv.py           # Testes unitários de WoE e IV
    ├── test_scorecard.py        # Testes de calibração FICO e predições
    ├── test_metrics.py          # Testes de validação de KS, ROC e Gini
    └── test_simulator.py        # Testes de monotonicidade do Cut-off
```

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- **Python 3.11** ou **Python 3.12**
- Git instalado

### 1. Clonar o repositório
```bash
git clone https://github.com/Renanbritto/credit-scoring-model.git
cd credit-scoring-model
```

### 2. Criar e ativar o ambiente virtual
```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# Linux / MacOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Executar os Testes Automatizados e Linter
```bash
# Execução dos testes
pytest -v tests/

# Checagem de qualidade de código
ruff check .
```

### 5. Iniciar o Dashboard Streamlit
```bash
streamlit run app.py
```
O navegador abrirá automaticamente em `http://localhost:8501`.

---

## 📊 Principais Resultados do Modelo

| Indicador Estatístico / Financeiro | Resultado Obtido | Padrão da Indústria |
|---|---|---|
| **Estatística Kolmogorov-Smirnov ($K\text{-}S$)** | **48.6%** | $> 40.0\%$ (Excelente) |
| **Área sob a Curva ROC ($\text{AUC}$)** | **0.842** | $> 0.750$ (Forte) |
| **Coeficiente Gini** | **68.4%** | $> 50.0\%$ (Alto poder preditivo) |
| **Taxa de Aprovação (Cut-off 540)** | **72.5%** | Alinhado ao apetite de risco |
| **Bad Rate da Safra Aprovada** | **3.8%** | Redução de 72% vs base bruta (13.5%) |
| **Perdas Evitadas Estimadas** | **> R$ 4.200.000** | Proponentes de alto risco barrados |

---

## 👤 Autor

**Renan Nocelli**  
Analista de Dados & Especialista em Modelagem Estatística e Engenharia de Risco Financeiro.

- **Portfólio Interativo:** [renan-nocelli.vercel.app](https://renan-nocelli.vercel.app)
- **LinkedIn:** [linkedin.com/in/renannocelli](https://www.linkedin.com/in/renannocelli)
- **GitHub:** [github.com/Renanbritto](https://github.com/Renanbritto)

---
*Licença MIT. Livre para uso educacional e profissional com citação de autoria.*
