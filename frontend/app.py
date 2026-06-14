# frontend/app.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests
import streamlit as st

from backend.agent import (
    dt_candidates,
    dt_candidates_proporcao,
    dt_children,
    dt_is_leaf,
    dt_question,
    dt_walk,
    feature_columns,
    get_next_feature_nb,
    nb,
)

# ─── Caminhos ─────────────────────────────────────────────────────────────────

ASSETS_DIR = Path(__file__).parent / "assets"
LOGO = str(ASSETS_DIR / "pokinator.png")
ICON = str(ASSETS_DIR / "Poké_Ball_icon.svg.png")

ALGO_DT = "Árvore de Decisão"
ALGO_NB = "Naive Bayes"

# ─── Mapeamento de perguntas amigáveis ────────────────────────────────────────

PERGUNTAS_AMIGAVEIS = {
    # tamanho
    "tamanho_pequeno": "O Pokémon que você pensou é pequeno?",
    "tamanho_médio": "O Pokémon que você pensou é de tamanho médio?",
    "tamanho_grande": "O Pokémon que você pensou é grande?",
    # pesado
    "pesado": "O Pokémon que você pensou é pesado (acima de 10kg)?",
    # tipo
    "tipo_fogo": "O Pokémon que você pensou é do tipo Fogo?",
    "tipo_água": "O Pokémon que você pensou é do tipo Água?",
    "tipo_planta": "O Pokémon que você pensou é do tipo Planta?",
    "tipo_elétrico": "O Pokémon que você pensou é do tipo Elétrico?",
    "tipo_gelo": "O Pokémon que você pensou é do tipo Gelo?",
    "tipo_lutador": "O Pokémon que você pensou é do tipo Lutador?",
    "tipo_veneno": "O Pokémon que você pensou é do tipo Veneno?",
    "tipo_terra": "O Pokémon que você pensou é do tipo Terra?",
    "tipo_voador": "O Pokémon que você pensou é do tipo Voador?",
    "tipo_psíquico": "O Pokémon que você pensou é do tipo Psíquico?",
    "tipo_inseto": "O Pokémon que você pensou é do tipo Inseto?",
    "tipo_pedra": "O Pokémon que você pensou é do tipo Pedra?",
    "tipo_fantasma": "O Pokémon que você pensou é do tipo Fantasma?",
    "tipo_dragão": "O Pokémon que você pensou é do tipo Dragão?",
    "tipo_sombrio": "O Pokémon que você pensou é do tipo Sombrio?",
    "tipo_aço": "O Pokémon que você pensou é do tipo Aço?",
    "tipo_fada": "O Pokémon que você pensou é do tipo Fada?",
    "tipo_normal": "O Pokémon que você pensou é do tipo Normal?",
    # cor
    "cor_vermelho": "O Pokémon que você pensou é predominantemente vermelho?",
    "cor_azul": "O Pokémon que você pensou é predominantemente azul?",
    "cor_amarelo": "O Pokémon que você pensou é predominantemente amarelo?",
    "cor_verde": "O Pokémon que você pensou é predominantemente verde?",
    "cor_preto": "O Pokémon que você pensou é predominantemente preto?",
    "cor_branco": "O Pokémon que você pensou é predominantemente branco?",
    "cor_marrom": "O Pokémon que você pensou é predominantemente marrom?",
    "cor_cinza": "O Pokémon que você pensou é predominantemente cinza?",
    "cor_roxo": "O Pokémon que você pensou é predominantemente roxo?",
    "cor_rosa": "O Pokémon que você pensou é predominantemente rosa?",
    # habitat
    "habitat_floresta": "O Pokémon que você pensou vive em florestas?",
    "habitat_caverna": "O Pokémon que você pensou vive em cavernas?",
    "habitat_mar": "O Pokémon que você pensou vive no mar?",
    "habitat_montanha": "O Pokémon que você pensou vive em montanhas?",
    "habitat_campo": "O Pokémon que você pensou vive em campos abertos?",
    "habitat_urbano": "O Pokémon que você pensou vive em áreas urbanas?",
    "habitat_beira_dagua": "O Pokémon que você pensou vive perto de rios ou lagos?",
    "habitat_terreno_acidentado": "O Pokémon que você pensou vive em terrenos acidentados?",
    "habitat_raro": "O Pokémon que você pensou é raro ou difícil de encontrar?",
    # forma
    "forma_bípede": "O Pokémon que você pensou fica em pé sobre duas pernas, tem braços, e possui cauda?",
    "forma_quadrúpede": "O Pokémon que você pensou anda sobre quatro patas?",
    "forma_alado_não_inseto": "O Pokémon que você pensou não é um inseto e tem asas (pássaros, morcegos, dinossauros)?",
    "forma_inseto_alado": "O Pokémon que você pensou parece um inseto com asas?",
    "forma_aquático": "O Pokémon que você pensou tem formato de peixe?",
    "forma_serpentiforme": "O Pokémon que você pensou tem formato de cobra ou serpente?",
    "forma_humanoide": "O Pokémon que você pensou fica em pé sobre duas pernas, tem braços, mas não possui cauda?",
    "forma_esférico": "O Pokémon que você pensou tem formato esférico ou arredondado?",
    "forma_amorfo": "O Pokémon que você pensou não tem uma forma definida?",
    "forma_artropode": "O Pokémon que você pensou parece um artrópode (caranguejo, aranha...)?",
    "forma_corpo_com_braços": "O Pokémon que você pensou tem braços, mas não tem pernas?",
    "forma_tentáculos": "O Pokémon que você pensou tem tentáculos?",
    "forma_múltiplas_cabeças": "O Pokémon que você pensou tem múltiplas cabeças?",
    "forma_corpo_com_pernas": "O Pokémon que você pensou tem pernas, mas não tem braços?",
    # estágio evolutivo
    "estagio_inicial": "O Pokémon que você pensou é a forma inicial da evolução?",
    "estagio_médio": "O Pokémon que você pensou é a forma intermediária da evolução?",
    "estagio_final": "O Pokémon que você pensou é a forma final da evolução?",
    "estagio_único": "O Pokémon que você pensou não evolui?",
    # geração
    "geracao_1": "O Pokémon que você pensou é da 1ª geração (Kanto)?",
    "geracao_2": "O Pokémon que você pensou é da 2ª geração (Johto)?",
    "geracao_3": "O Pokémon que você pensou é da 3ª geração (Hoenn)?",
    "geracao_4": "O Pokémon que você pensou é da 4ª geração (Sinnoh)?",
    "geracao_5": "O Pokémon que você pensou é da 5ª geração (Unova)?",
    "geracao_6": "O Pokémon que você pensou é da 6ª geração (Kalos)?",
    "geracao_7": "O Pokémon que você pensou é da 7ª geração (Alola)?",
    "geracao_8": "O Pokémon que você pensou é da 8ª geração (Galar)?",
    "geracao_9": "O Pokémon que você pensou é da 9ª geração (Paldea)?",
}


def pergunta_amigavel(feature):
    # Para features não mapeadas, exibe o nome da feature como fallback
    return PERGUNTAS_AMIGAVEIS.get(feature, feature)


# ─── Estado do jogo ───────────────────────────────────────────────────────────

def init_game():
    """Reinicia o estado da partida no session_state (vale para os dois algoritmos)."""
    st.session_state.node = 0
    st.session_state.path = [0]
    st.session_state.answers = {}
    st.session_state.probas = np.ones(len(nb.classes_)) / len(nb.classes_)
    st.session_state.perguntas = 0
    st.session_state.current_feature = None
    st.session_state.candidatos = []
    st.session_state.guess_idx = 0
    st.session_state.game_over = False


# ─── Árvore de Decisão ────────────────────────────────────────────────────────

def step_dt():
    """Avança a árvore: define a próxima pergunta ou encerra com os candidatos da folha."""
    node = st.session_state.node
    if dt_is_leaf(node):
        st.session_state.candidatos = dt_candidates(node)
        st.session_state.game_over = True
    else:
        st.session_state.current_feature = dt_question(node)


def answer_dt(resp):  # resp: True (sim) / False (não)
    """Registra a resposta e desce para o nó filho correspondente da árvore."""
    st.session_state.node = dt_walk(st.session_state.node, resp)
    st.session_state.path.append(st.session_state.node)
    st.session_state.perguntas += 1


# ─── Naive Bayes ──────────────────────────────────────────────────────────────

def finalizar_nb():
    """Encerra a partida do NB com os candidatos ordenados por probabilidade."""
    probas = st.session_state.probas
    indices = np.argsort(probas)[::-1]
    st.session_state.candidatos = [nb.classes_[i] for i in indices if probas[i] > 0]
    st.session_state.game_over = True


def step_nb():
    """Define a próxima pergunta do NB ou finaliza ao atingir 80% de certeza."""
    probas = st.session_state.probas

    if probas.max() >= 0.8:
        finalizar_nb()
        return

    feature = get_next_feature_nb(st.session_state.answers, probas)
    if feature is None:
        # Sem mais perguntas úteis: usa os melhores palpites
        finalizar_nb()
        return

    st.session_state.current_feature = feature


def answer_nb(resp):  # resp: "s" / "n" / "ns"
    """Registra a resposta e atualiza as probabilidades das classes (regra de Bayes)."""
    feature = st.session_state.current_feature
    st.session_state.perguntas += 1

    if resp == "ns":
        st.session_state.answers[feature] = None
        return

    valor = resp == "s"
    st.session_state.answers[feature] = valor

    probas = st.session_state.probas
    feature_idx = feature_columns.index(feature)
    for i in range(len(nb.classes_)):
        theta = nb.theta_[i, feature_idx]
        likelihood = theta if valor else 1 - theta
        probas[i] *= likelihood

    total = probas.sum()
    if total > 0:
        probas /= total
    st.session_state.probas = probas


# ─── Detalhes do modelo (expansíveis) ─────────────────────────────────────────

def rotulo_no(feature):
    # Rótulo curto para os nós da árvore (ex.: "tipo_fogo" → "Tipo fogo?")
    return feature.replace("_", " ").capitalize() + "?"


def get_pokemons_restantes_dt():
    """Pokémons ainda possíveis no nó atual da árvore, com proporção em cada um."""
    return [
        (formatar_nome(nome), proporcao)
        for nome, proporcao in dt_candidates_proporcao(st.session_state.node)
    ]


def build_subtree_dot(start_node, max_depth=3):
    """Gera um grafo DOT da subárvore a partir do nó atual, com profundidade limitada."""
    lines = [
        "digraph Tree {",
        'node [shape=box, style="rounded,filled", fillcolor="#f0f2f6", fontname="sans-serif", fontsize=11];',
        'edge [fontname="sans-serif", fontsize=10];',
    ]

    def add_node(node, depth):
        if dt_is_leaf(node):
            classe = formatar_nome(dt_candidates(node)[0])
            lines.append(f'{node} [label="{classe}", fillcolor="#d4edda"];')
            return

        if depth >= max_depth:
            restantes = len(dt_candidates(node))
            lines.append(f'{node} [label="… {restantes} Pokémon(s)", fillcolor="#fff3cd"];')
            return

        feature = dt_question(node)
        cor = "#cfe2ff" if node == start_node else "#f0f2f6"
        lines.append(f'{node} [label="{rotulo_no(feature)}", fillcolor="{cor}"];')

        esq, dir = dt_children(node)
        add_node(esq, depth + 1)
        add_node(dir, depth + 1)
        lines.append(f'{node} -> {esq} [label="Não"];')
        lines.append(f'{node} -> {dir} [label="Sim"];')

    add_node(start_node, 0)
    lines.append("}")
    return "\n".join(lines)


def build_path_dot():
    """Gera um grafo DOT do caminho percorrido na árvore, da raiz até o palpite."""
    path = st.session_state.path
    lines = [
        "digraph Tree {",
        'node [shape=box, style="rounded,filled", fillcolor="#f0f2f6", fontname="sans-serif", fontsize=11];',
        'edge [fontname="sans-serif", fontsize=10];',
    ]

    for i, node in enumerate(path):
        if dt_is_leaf(node):
            # Mostra o palpite atual (pode mudar se o usuário rejeitar os anteriores)
            candidatos = st.session_state.candidatos
            idx = min(st.session_state.guess_idx, len(candidatos) - 1)
            classe = formatar_nome(candidatos[idx])
            lines.append(f'{node} [label="{classe}", fillcolor="#d4edda"];')
        else:
            feature = dt_question(node)
            lines.append(f'{node} [label="{rotulo_no(feature)}"];')

        if i > 0:
            anterior = path[i - 1]
            _, direita = dt_children(anterior)
            resp = "Sim" if node == direita else "Não"
            lines.append(f'{anterior} -> {node} [label="{resp}"];')

    lines.append("}")
    return "\n".join(lines)


def get_pokemons_provaveis_nb(n=10):
    """Top N Pokémons mais prováveis segundo o Naive Bayes."""
    probas = st.session_state.probas
    indices = np.argsort(probas)[::-1][:n]
    return [
        (formatar_nome(nb.classes_[i]), float(probas[i]))
        for i in indices
        if probas[i] > 0
    ]


def render_tabela_probabilidades(dados, label):
    """Renderiza uma tabela de Pokémon com barra de progresso na coluna de valor."""
    df = pd.DataFrame(dados, columns=["Pokémon", label])
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            label: st.column_config.ProgressColumn(
                label, min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )


def get_respostas_dadas():
    """Lista (feature, resposta) das perguntas já respondidas, na ordem."""
    if st.session_state.algorithm == ALGO_DT:
        path = st.session_state.path
        return [
            (dt_question(anterior), node == dt_children(anterior)[1])
            for anterior, node in zip(path, path[1:])
        ]
    return list(st.session_state.answers.items())


def render_respostas():
    """Expander com as características já respondidas na partida."""
    with st.expander("Ver características já definidas"):
        respostas = get_respostas_dadas()
        if not respostas:
            st.caption("Nenhuma pergunta respondida ainda.")
            return

        rotulos = {True: "Sim", False: "Não", None: "Não sei"}
        df = pd.DataFrame(
            [(rotulo_no(feature), rotulos[resp]) for feature, resp in respostas],
            columns=["Característica", "Resposta"],
        )
        st.dataframe(df, hide_index=True, use_container_width=True)


def render_detalhes():
    """Expander com a visualização do modelo: grafo da árvore ou ranking do NB."""
    game_over = st.session_state.game_over

    if st.session_state.algorithm == ALGO_DT:
        if game_over:
            with st.expander("Ver caminho percorrido na árvore"):
                st.caption("Caminho da raiz até o palpite, com as respostas dadas.")
                st.graphviz_chart(build_path_dot(), use_container_width=True)

                restantes = get_pokemons_restantes_dt()
                st.markdown(f"**Pokémons na folha final ({len(restantes)}):**")
                render_tabela_probabilidades(restantes, "Proporção")
        else:
            with st.expander("Ver árvore de decisão e Pokémons restantes"):
                st.caption(
                    "Trecho da árvore a partir da pergunta atual (nó azul). "
                    "Os nós amarelos indicam ramos com mais perguntas adiante."
                )
                st.graphviz_chart(
                    build_subtree_dot(st.session_state.node), use_container_width=True
                )

                restantes = get_pokemons_restantes_dt()
                st.markdown(f"**Pokémons restantes ({len(restantes)}):**")
                render_tabela_probabilidades(restantes, "Proporção")
    else:
        with st.expander("Ver Pokémons mais prováveis"):
            render_tabela_probabilidades(
                get_pokemons_provaveis_nb(), "Probabilidade"
            )


# ─── PokéAPI ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_official_artwork(nome):
    """Busca na PokéAPI a URL da arte oficial do Pokémon (resultado em cache)."""
    resp = requests.get(f"https://pokeapi.co/api/v2/pokemon/{nome}", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["sprites"]["other"]["official-artwork"]["front_default"]


def formatar_nome(nome):
    """Formata o nome para exibição (ex.: "mr-mime" -> "Mr Mime")."""
    return nome.replace("-", " ").title()


# ─── Renderização ─────────────────────────────────────────────────────────────

def render_centered_image(src, **kwargs):
    """Exibe uma imagem centralizada usando uma coluna do meio mais larga."""
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.image(src, **kwargs)


def render_question():
    """Mostra a pergunta atual e os botões de resposta (3 no NB, 2 na árvore)."""
    feature = st.session_state.current_feature

    st.markdown(
        f"<h3 style='text-align:center'>{pergunta_amigavel(feature)}</h3>",
        unsafe_allow_html=True,
    )

    is_nb = st.session_state.algorithm == ALGO_NB
    cols = st.columns(3 if is_nb else 2)

    if is_nb:
        cols[0].button("Sim", use_container_width=True, on_click=answer_nb, args=("s",))
        cols[1].button("Não", use_container_width=True, on_click=answer_nb, args=("n",))
        # "Não sei" só aparece no Naive Bayes
        cols[2].button("Não sei", use_container_width=True, on_click=answer_nb, args=("ns",))
    else:
        cols[0].button("Sim", use_container_width=True, on_click=answer_dt, args=(True,))
        cols[1].button("Não", use_container_width=True, on_click=answer_dt, args=(False,))


def proximo_palpite():
    """Passa para o próximo candidato quando o usuário rejeita o palpite atual."""
    st.session_state.guess_idx += 1


def render_result():
    """Mostra o palpite atual (com a arte do Pokémon) e os botões de continuar/reiniciar."""
    candidatos = st.session_state.candidatos
    idx = st.session_state.guess_idx

    if idx >= len(candidatos):
        st.warning(
            "Você me venceu! Não conheço outro Pokémon com essas características."
        )
        st.button("Jogar novamente", use_container_width=True, on_click=init_game)
        return

    nome = candidatos[idx]
    caption = formatar_nome(nome)

    try:
        url = fetch_official_artwork(nome)
        if url:
            render_centered_image(url, caption=caption, use_container_width=True)
    except Exception:
        # Sem imagem em caso de erro na PokéAPI
        pass

    if idx == 0:
        perguntas = st.session_state.perguntas
        st.success(f"É o **{caption}**! Acertei em {perguntas} pergunta(s).")
    else:
        st.success(f"Hmm... então só pode ser o **{caption}**!")

    cols = st.columns(2)
    cols[0].button("Não é esse", use_container_width=True, on_click=proximo_palpite)
    cols[1].button("Jogar novamente", use_container_width=True, on_click=init_game)


# ─── App ──────────────────────────────────────────────────────────────────────

def main():
    """Monta a página: sidebar, logo, avança o estado do jogo e renderiza a tela."""
    page_icon = ICON if Path(ICON).exists() else None
    st.set_page_config(page_title="Pokinator", page_icon=page_icon)

    # Sidebar
    st.sidebar.title("Escolha o algoritmo")
    algorithm = st.sidebar.radio(
        "Algoritmo", [ALGO_DT, ALGO_NB], label_visibility="collapsed"
    )

    # Troca de algoritmo reinicia o jogo
    if "algorithm" not in st.session_state or st.session_state.algorithm != algorithm:
        st.session_state.algorithm = algorithm
        init_game()

    if st.sidebar.button("Reiniciar", use_container_width=True):
        init_game()

    # Logo centralizada no topo
    render_centered_image(LOGO, use_container_width=True)

    # Avança o estado do jogo (define a pergunta atual ou o resultado)
    if not st.session_state.game_over:
        if st.session_state.algorithm == ALGO_DT:
            step_dt()
        else:
            step_nb()

    if st.session_state.game_over:
        render_result()
    else:
        render_question()
    render_respostas()
    render_detalhes()


if __name__ == "__main__":
    main()
