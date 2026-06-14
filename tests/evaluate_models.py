# tests/evaluate_models.py
"""Avaliação dos modelos do Pokinator sobre os dados processados.

Para cada Pokémon, simula uma partida respondendo a cada pergunta com o valor
real da feature (sempre com a verdade, nunca "não sei") e mede:

  * acurácia top-1 — em quantos Pokémon o primeiro palpite do modelo está certo;
  * número de perguntas até o palpite (média, mediana e máximo).

A lógica de jogo é a mesma do backend/frontend, então os números refletem o que
o usuário veria na prática.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backend.agent import (
    dt_candidates,
    dt_is_leaf,
    dt_question,
    dt_walk,
    feature_columns,
    get_next_feature_nb,
    nb,
)

DATA = Path(__file__).parent.parent / "data" / "pokemon_processed.csv"


def simulate_decision_tree(features):
    """Percorre a árvore até uma folha e retorna (palpite, nº de perguntas).

    Em cada nó interno, "sim" (feature verdadeira) vai para o filho direito e
    "não" para o esquerdo — a mesma convenção usada no app.
    """
    node = 0
    perguntas = 0
    while not dt_is_leaf(node):
        node = dt_walk(node, features[dt_question(node)])
        perguntas += 1
    return dt_candidates(node)[0], perguntas


def simulate_naive_bayes(features):
    """Joga uma partida de Naive Bayes e retorna (palpite, nº de perguntas).

    Repete a escolha de pergunta de get_next_feature_nb, atualizando as
    probabilidades a cada resposta, até atingir 80% de certeza ou esgotar as
    perguntas úteis (mesma condição de parada do app).
    """
    answers = {}
    probas = np.ones(len(nb.classes_)) / len(nb.classes_)
    perguntas = 0

    while True:
        best_idx = probas.argmax()
        if probas[best_idx] >= 0.8:
            break

        feature = get_next_feature_nb(answers, probas)
        if feature is None:  # sem perguntas úteis: fica com o melhor palpite
            break

        valor = bool(features[feature])
        answers[feature] = valor
        perguntas += 1

        feature_idx = feature_columns.index(feature)
        theta = nb.theta_[:, feature_idx]
        probas *= theta if valor else 1 - theta
        total = probas.sum()
        if total > 0:
            probas /= total

    return nb.classes_[best_idx], perguntas


def evaluate(df, simulate):
    """Roda `simulate` para cada Pokémon e agrega acertos e perguntas."""
    acertos = 0
    perguntas = []
    for _, row in df.iterrows():
        features = row.drop(["id", "name"]).to_dict()
        palpite, n = simulate(features)
        acertos += palpite == row["name"]
        perguntas.append(n)

    perguntas = np.array(perguntas)
    total = len(df)
    return {
        "acertos": acertos,
        "total": total,
        "acuracia": acertos / total,
        "perguntas_media": perguntas.mean(),
        "perguntas_mediana": np.median(perguntas),
        "perguntas_max": perguntas.max(),
    }


def find_indistinguishable_pairs(df):
    """Grupos de Pokémon com vetores de features idênticos.

    Nenhum modelo consegue distinguir Pokémon com features iguais, então esses
    grupos definem o teto teórico de acerto: um por grupo fica inalcançável.
    """
    features = df.drop(columns=["id", "name"])
    duplicados = df[features.duplicated(keep=False)]
    grupos = duplicados.groupby(list(features.columns)).groups
    return [list(df.loc[idx, "name"]) for idx in grupos.values()]


def print_report(nome, r):
    print(f"\n=== {nome} ===")
    print(f"  Acertos (top-1):    {r['acertos']}/{r['total']} ({r['acuracia']:.1%})")
    print(f"  Perguntas (média):  {r['perguntas_media']:.2f}")
    print(f"  Perguntas (mediana):{r['perguntas_mediana']:.0f}")
    print(f"  Perguntas (máximo): {r['perguntas_max']}")


def main():
    df = pd.read_csv(DATA)
    total = len(df)
    print(f"Avaliando {total} Pokémon dos dados processados...")

    # Teto teórico: cada grupo de features idênticas perde, no melhor caso, todos
    # os Pokémon menos um.
    pares = find_indistinguishable_pairs(df)
    inalcancaveis = sum(len(g) - 1 for g in pares)
    print(f"\n{len(pares)} grupo(s) de Pokémon com features idênticas "
          f"(teto de acerto: {total - inalcancaveis}/{total}):")
    for grupo in pares:
        print(f"  - {', '.join(grupo)}")

    print_report("Árvore de Decisão", evaluate(df, simulate_decision_tree))
    print_report("Naive Bayes", evaluate(df, simulate_naive_bayes))


if __name__ == "__main__":
    main()
