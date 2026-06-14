# backend/agent.py
"""Backend do Pokinator: carrega os modelos treinados e fornece a lógica de jogo
(Naive Bayes e Árvore de Decisão) usada pela interface (frontend/app.py)."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import _tree

MODELS_DIR = Path(__file__).parent.parent / "models"

# Artefatos gerados na etapa de treino (ver notebooks/).
feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
dt = joblib.load(MODELS_DIR / "decision_tree.pkl")
nb = joblib.load(MODELS_DIR / "naive_bayes.pkl")

# ─── Naive Bayes ──────────────────────────────────────────────────────────────

# Grupos de features mutuamente exclusivas: um Pokémon tem só um valor por grupo
# (uma cor, um estágio, ...), então um "sim" dentro do grupo o encerra.
exclusive_groups = {
    "tamanho": [c for c in feature_columns if c.startswith("tamanho_")],
    "estagio": [c for c in feature_columns if c.startswith("estagio_")],
    "cor": [c for c in feature_columns if c.startswith("cor_")],
    "forma": [c for c in feature_columns if c.startswith("forma_")],
    "geracao": [c for c in feature_columns if c.startswith("geracao_")],
    "habitat": [c for c in feature_columns if c.startswith("habitat_")],
}


def get_group(feature):
    """Retorna o grupo exclusivo da feature, ou None se ela não pertence a nenhum."""
    for group in exclusive_groups:
        if feature.startswith(f"{group}_"):
            return group
    return None


def get_answered_groups(answers):
    """Grupos que já receberam um "sim" e não precisam de mais perguntas."""
    answered = set()
    for group, cols in exclusive_groups.items():
        if any(answers.get(col) is True for col in cols):
            answered.add(group)
    return answered


def get_next_feature_nb(answers, probas):
    """Escolhe a próxima pergunta do Naive Bayes.

    Prefere a feature que melhor separa os candidatos ainda possíveis: aquela
    cuja probabilidade média (entre as classes com chance > 0) é mais próxima de
    0.5. Ignora features já respondidas e grupos exclusivos já resolvidos.
    """
    answered_features = set(answers.keys())
    answered_groups = get_answered_groups(answers)

    # Índices das classes que ainda estão em jogo.
    top_indices = np.where(probas > 0)[0]

    best_feature = None
    best_score = -1

    for i, feature in enumerate(feature_columns):
        if feature in answered_features:
            continue
        group = get_group(feature)
        if group and group in answered_groups:
            continue

        mean = np.mean(nb.theta_[top_indices, i])
        score = 1 - abs(mean - 0.5) * 2  # 1 quando média == 0.5; 0 nos extremos
        if score > best_score:
            best_score = score
            best_feature = feature

    return best_feature


# ─── Árvore de Decisão ────────────────────────────────────────────────────────
#
# A árvore é percorrida por nó: cada nó interno faz uma pergunta e, conforme a
# resposta, segue para um dos filhos até chegar a uma folha. Por convenção do
# sklearn, "sim" (feature verdadeira) vai para a direita e "não" para a esquerda.


def dt_is_leaf(node):
    """True se o nó é uma folha (não há mais perguntas, só o palpite)."""
    return dt.tree_.children_left[node] == _tree.TREE_LEAF


def dt_question(node):
    """Feature perguntada em um nó interno."""
    return feature_columns[dt.tree_.feature[node]]


def dt_children(node):
    """Filhos de um nó interno como (esquerda = "não", direita = "sim")."""
    return dt.tree_.children_left[node], dt.tree_.children_right[node]


def dt_walk(node, resp):
    """Avança para o próximo nó dada a resposta (True = sim → direita)."""
    esquerda, direita = dt_children(node)
    return direita if resp else esquerda


def dt_candidates(node):
    """Pokémon possíveis no nó, ordenados da maior para a menor proporção."""
    valores = dt.tree_.value[node][0]
    return [dt.classes_[i] for i in np.argsort(valores)[::-1] if valores[i] > 0]


def dt_candidates_proporcao(node):
    """(classe, proporção) dos Pokémon possíveis no nó, do mais ao menos provável."""
    valores = dt.tree_.value[node][0]
    total = valores.sum()
    indices = np.argsort(valores)[::-1]
    return [(dt.classes_[i], valores[i] / total) for i in indices if valores[i] > 0]
