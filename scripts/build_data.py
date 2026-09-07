"""
build_data.py
Script para gerar o dataset sintético e o dicionário de regras do Scorecard.
"""

import os
import sys

# Garante que a raiz do projeto esteja no PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator import generate_credit_data
from src.scorecard import ScorecardModel


def main():
    print("Gerando 45.000 proponentes...")
    df = generate_credit_data(n_samples=45000, seed=42)

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    sample_path = "data/raw/credit_applications_sample.csv"
    df.head(5000).to_csv(sample_path, index=False)
    print(f"Amostra de 5.000 linhas salva em {sample_path}")

    print("Ajustando modelo de Scorecard FICO...")
    model = ScorecardModel()
    model.fit(df, target_col="default")

    rules_df = model.get_scorecard_dataframe()
    dict_path = "data/processed/scorecard_dictionary.csv"
    rules_df.to_csv(dict_path, index=False)
    print(f"Dicionário de regras salvo em {dict_path}")
    print("Concluído com sucesso!")


if __name__ == "__main__":
    main()
