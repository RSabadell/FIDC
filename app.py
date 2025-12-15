
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

# Configuração básica da página
st.set_page_config(
    page_title="Dashboard de Crédito",
    layout="wide",
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Dashboard_Credito_BI.csv", sep=";", encoding='utf-8-sig')
    except:
        df = pd.read_csv("Dashboard_Credito_BI.csv", encoding='utf-8-sig')

    # Criar colunas derivadas se não existirem
    if 'Regiao' not in df.columns and 'Estado' in df.columns:
        regioes = {
            'Norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'],
            'Nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
            'Centro-Oeste': ['DF', 'GO', 'MT', 'MS'],
            'Sudeste': ['ES', 'MG', 'RJ', 'SP'],
            'Sul': ['PR', 'RS', 'SC']
        }
        def map_regiao(estado):
            for k, v in regioes.items():
                if estado in v:
                    return k
            return 'Outros'
        df['Regiao'] = df['Estado'].apply(map_regiao)

    if 'Perfil_Risco' not in df.columns and 'Score SERASA' in df.columns:
        def calcular_perfil(score):
            if pd.isna(score):
                return 'Não informado'
            if score >= 700:
                return 'Baixo Risco'
            elif score >= 500:
                return 'Médio Risco'
            else:
                return 'Alto Risco'
        df['Perfil_Risco'] = df['Score SERASA'].apply(calcular_perfil)

    if 'Faixa Score' not in df.columns and 'Score SERASA' in df.columns:
        df['Faixa Score'] = pd.cut(
            df['Score SERASA'],
            bins=[0, 300, 500, 700, 900, 1000],
            labels=['0-300', '301-500', '501-700', '701-900', '901-1000']
        )

    if 'Faixa Idade' not in df.columns and 'Idade' in df.columns:
        df['Faixa Idade'] = pd.cut(
            df['Idade'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56+']
        )

    if 'Aprovado_Flag' not in df.columns and 'Status' in df.columns:
        df['Aprovado_Flag'] = df['Status'].isin(['Aprovado', 'Contratado']).astype(int)

    if 'Limite de Crédito' not in df.columns:
        df['Limite de Crédito'] = np.nan
    if 'Renda Pres' not in df.columns:
        df['Renda Pres'] = 0

    return df

try:
    df = load_data()
    st.sidebar.success("✅ Dados carregados com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {str(e)}")
    st.info("Certifique-se de que o arquivo 'Dashboard_Credito_BI.csv' está na mesma pasta do app.")
    st.stop()

st.title("📊 Dashboard Interativo - Portfólio de Crédito")

colunas_essenciais = ['Score SERASA', 'Idade', 'Status', 'Valor Financiado']
colunas_faltantes = [col for col in colunas_essenciais if col not in df.columns]

if colunas_faltantes:
    st.error(f"❌ Colunas faltando no arquivo: {', '.join(colunas_faltantes)}")
    st.stop()

# ===================== SIDEBAR – FILTROS =====================

st.sidebar.header("🔍 Filtros")

if 'Status' in df.columns:
    status_opts = sorted(df["Status"].dropna().unique())
    status_sel = st.sidebar.multiselect(
        "Status",
        options=status_opts,
        default=status_opts,
    )
else:
    status_sel = []

if 'Regiao' in df.columns:
    regiao_opts = sorted(df["Regiao"].dropna().unique())
    regiao_sel = st.sidebar.multiselect(
        "Região",
        options=regiao_opts,
        default=regiao_opts,
    )
else:
    regiao_sel = []

if 'Faixa Score' in df.columns:
    faixa_score_opts = sorted([str(x) for x in df["Faixa Score"].dropna().unique()])
    faixa_score_sel = st.sidebar.multiselect(
        "Faixa de Score",
        options=faixa_score_opts,
        default=faixa_score_opts,
    )
else:
    faixa_score_sel = []

if 'Faixa Idade' in df.columns:
    faixa_idade_opts = sorted([str(x) for x in df["Faixa Idade"].dropna().unique()])
    faixa_idade_sel = st.sidebar.multiselect(
        "Faixa de Idade",
        options=faixa_idade_opts,
        default=faixa_idade_opts,
    )
else:
    faixa_idade_sel = []

# Aplicar filtros
df_filt = df.copy()

if status_sel:
    df_filt = df_filt[df_filt["Status"].isin(status_sel)]
if regiao_sel and 'Regiao' in df.columns:
    df_filt = df_filt[df_filt["Regiao"].isin(regiao_sel)]
if faixa_score_sel and 'Faixa Score' in df.columns:
    df_filt = df_filt[df_filt["Faixa Score"].astype(str).isin(faixa_score_sel)]
if faixa_idade_sel and 'Faixa Idade' in df.columns:
    df_filt = df_filt[df_filt["Faixa Idade"].astype(str).isin(faixa_idade_sel)]

st.sidebar.markdown("---")
st.sidebar.write(f"Registros filtrados: **{len(df_filt)}** de **{len(df)}**")

# ===================== KPIs =====================

st.header("📈 Indicadores principais")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total de Propostas", f"{len(df_filt):,}".replace(",", "."))

with c2:
    if 'Aprovado_Flag' in df_filt.columns:
        taxa_aprov = (
            df_filt["Aprovado_Flag"].sum() / len(df_filt) * 100
            if len(df_filt) > 0 else 0
        )
        st.metric("Taxa de Aprovação", f"{taxa_aprov:.1f}%")
    else:
        st.metric("Taxa de Aprovação", "N/A")

with c3:
    if 'Score SERASA' in df_filt.columns:
        score_med = df_filt["Score SERASA"].mean() if len(df_filt) > 0 else 0
        st.metric("Score Médio", f"{score_med:.0f}")
    else:
        st.metric("Score Médio", "N/A")

with c4:
    if 'Valor Financiado' in df_filt.columns:
        valor_tot = df_filt["Valor Financiado"].sum() if len(df_filt) > 0 else 0
        st.metric("Valor Total", f"R$ {valor_tot:,.0f}".replace(",", "."))
    else:
        st.metric("Valor Total", "N/A")

# ===================== RIDGE PLOT POR ESTADO =====================

st.header("🏔️ Ridge Plot - Distribuição de Score por Estado")

st.info("💡 Estados ordenados por score médio. Cores indicam performance: vermelho (baixo) → verde (alto)")

if 'Estado' in df_filt.columns and 'Score SERASA' in df_filt.columns:

    # Configurações do ridge plot
    col_ridge1, col_ridge2, col_ridge3 = st.columns(3)

    with col_ridge1:
        min_obs_ridge = st.slider(
            "Mínimo de observações por estado",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
            help="Estados com menos observações serão excluídos"
        )

    with col_ridge2:
        var_ridge = st.selectbox(
            "Variável para Ridge Plot",
            ["Score SERASA", "Idade", "Valor Financiado"],
            help="Escolha qual variável analisar por estado"
        )

    with col_ridge3:
        tema_ridge = st.selectbox(
            "Tema de cores",
            ["Gradiente Performance", "Fundo Escuro", "Seaborn"],
            help="Estilo visual do gráfico"
        )

    # Preparar dados
    df_ridge = df_filt[['Estado', var_ridge]].dropna()

    # Filtrar estados com amostra mínima
    estados_contagem = df_ridge.groupby('Estado')[var_ridge].count()
    estados_validos = estados_contagem[estados_contagem >= min_obs_ridge].index

    df_ridge = df_ridge[df_ridge['Estado'].isin(estados_validos)]

    if len(estados_validos) < 3:
        st.warning(f"⚠️ Apenas {len(estados_validos)} estado(s) com mínimo de {min_obs_ridge} observações. Reduza o mínimo.")
    else:
        # Calcular média por estado e ordenar
        score_medio = df_ridge.groupby('Estado')[var_ridge].mean().sort_values()
        ordem_estados = score_medio.index
        n_estados = len(ordem_estados)

        # Normalizar valores se for Score (0-10 scale)
        if var_ridge == "Score SERASA":
            df_ridge['Valor_norm'] = df_ridge[var_ridge] / 100.0
            x_label = f'{var_ridge} (0-10)'
            x_max = 10
        else:
            df_ridge['Valor_norm'] = df_ridge[var_ridge]
            x_label = var_ridge
            x_max = df_ridge['Valor_norm'].max()

        # Criar figura
        if tema_ridge == "Fundo Escuro":
            fig_ridge, ax_ridge = plt.subplots(figsize=(12, max(8, n_estados * 0.4)))
            fig_ridge.patch.set_facecolor('#1a1a1a')
            ax_ridge.set_facecolor('#1a1a1a')
            text_color = 'white'
            grid_color = 'white'
            grid_alpha = 0.15
        else:
            fig_ridge, ax_ridge = plt.subplots(figsize=(12, max(8, n_estados * 0.4)))
            fig_ridge.patch.set_facecolor('white')
            ax_ridge.set_facecolor('white')
            text_color = 'black'
            grid_color = 'gray'
            grid_alpha = 0.3

        # Gradiente de cores
        if tema_ridge == "Seaborn":
            cores = sns.color_palette("husl", n_estados)
        else:
            colors_list = ['#8B0000', '#CD5C5C', '#FF6347', '#FF8C00', '#FFA500', 
                          '#FFD700', '#FFFF00', '#ADFF2F', '#7FFF00', '#00FF00', '#228B22']
            cmap = LinearSegmentedColormap.from_list('score_gradient', colors_list, N=n_estados)
            cores = [cmap(i/n_estados) for i in range(n_estados)]

        # Grid
        x_range = np.linspace(0, x_max, 500)
        spacing = 0.8

        # Plotar cada estado
        for i, (estado, cor) in enumerate(zip(ordem_estados, cores)):
            dados = df_ridge.loc[df_ridge['Estado'] == estado, 'Valor_norm'].dropna()

            if len(dados) < 5:
                continue

            # Calcular KDE
            try:
                kde = gaussian_kde(dados, bw_method=0.15)
                density = kde(x_range)
                density = density / density.max() * 0.7  # Normalizar altura
            except:
                continue

            y_base = i * spacing

            # Plotar ridge
            ax_ridge.fill_between(x_range, y_base, y_base + density, 
                                 color=cor, alpha=0.85, linewidth=0.5, 
                                 edgecolor=text_color if tema_ridge == "Fundo Escuro" else 'black')

            ax_ridge.plot(x_range, y_base + density, color=text_color, 
                         linewidth=0.3, alpha=0.6)

            # Labels
            media_estado = dados.mean() * (100 if var_ridge == "Score SERASA" else 1)
            ax_ridge.text(-0.3, y_base + 0.35, f'{estado}', 
                        va='center', ha='right', fontsize=9, 
                        color=text_color, weight='bold')
            ax_ridge.text(x_max + 0.3, y_base + 0.35, f'{media_estado:.0f}', 
                        va='center', ha='left', fontsize=8, color=text_color)

        # Configurar eixos
        ax_ridge.set_xlim(-0.5, x_max + 0.5)
        ax_ridge.set_ylim(-0.5, n_estados * spacing)

        # Grid vertical
        if var_ridge == "Score SERASA":
            for x in range(11):
                ax_ridge.axvline(x, color=grid_color, alpha=grid_alpha, 
                               linewidth=0.5, linestyle='-')
            ax_ridge.set_xticks(range(11))
        else:
            ax_ridge.grid(axis='x', alpha=grid_alpha)

        ax_ridge.set_xlabel(x_label, fontsize=11, color=text_color, weight='bold')
        ax_ridge.set_xticklabels(ax_ridge.get_xticks(), color=text_color, fontsize=9)
        ax_ridge.set_yticks([])

        for spine in ax_ridge.spines.values():
            spine.set_visible(False)

        ax_ridge.text(x_max/2, n_estados * spacing + 0.5, 
                    f'Distribuição de {var_ridge} por Estado', 
                    ha='center', fontsize=14, color=text_color, weight='bold')

        plt.tight_layout()
        st.pyplot(fig_ridge)

        # Estatísticas
        with st.expander("📊 Estatísticas por Estado"):
            stats_estado = df_ridge.groupby('Estado')[var_ridge].agg(['count', 'mean', 'median', 'std']).round(2)
            stats_estado.columns = ['Contagem', 'Média', 'Mediana', 'Desvio Padrão']
            stats_estado = stats_estado.sort_values('Média', ascending=False)
            st.dataframe(stats_estado, use_container_width=True)

else:
    st.warning("⚠️ Colunas 'Estado' e 'Score SERASA' necessárias para Ridge Plot")

# ===================== KDE CONFIGURÁVEL =====================

st.header("📉 Distribuições (KDE)")

st.info("💡 Desmarcando 'Normalizar', as curvas mantêm proporções reais.")

variaveis_disponiveis = []
for var in ["Score SERASA", "Idade", "Limite de Crédito", "Renda Pres", "Valor Financiado"]:
    if var in df_filt.columns:
        variaveis_disponiveis.append(var)

segmentacoes_disponiveis = []
for seg in ["Status", "Regiao", "Faixa Score", "Faixa Idade", "Perfil_Risco"]:
    if seg in df_filt.columns:
        segmentacoes_disponiveis.append(seg)

if not variaveis_disponiveis or not segmentacoes_disponiveis:
    st.warning("⚠️ Dados insuficientes para gerar gráficos KDE")
else:
    cl, cr = st.columns(2)

    with cl:
        var_x = st.selectbox(
            "Variável contínua para o eixo X",
            variaveis_disponiveis,
        )

    with cr:
        hue = st.selectbox(
            "Segmentar por (hue)",
            segmentacoes_disponiveis,
        )

    col_norm1, col_norm2 = st.columns([2, 1])
    with col_norm1:
        normalizar = st.checkbox(
            "Normalizar curvas (cada curva com área = 1)", 
            value=False,
            help="Quando desmarcado, mantém as proporções reais entre categorias"
        )

    with col_norm2:
        if normalizar:
            st.caption("🔄 Modo: Densidades normalizadas")
        else:
            st.caption("📊 Modo: Frequências reais")

    fig, ax = plt.subplots(figsize=(10, 5))

    categorias_plotadas = 0
    for cat in df_filt[hue].dropna().unique():
        serie = df_filt.loc[df_filt[hue] == cat, var_x].dropna()

        if len(serie) > 5:
            categorias_plotadas += 1
            if normalizar:
                sns.kdeplot(
                    serie,
                    label=f"{cat} (n={len(serie)})",
                    fill=True,
                    alpha=0.25,
                    linewidth=1.5,
                    ax=ax,
                )
            else:
                try:
                    kde = stats.gaussian_kde(serie)
                    x_range = np.linspace(serie.min(), serie.max(), 200)
                    density = kde(x_range)
                    frequency = density * len(serie)

                    ax.plot(x_range, frequency, label=f"{cat} (n={len(serie)})", linewidth=1.5)
                    ax.fill_between(x_range, frequency, alpha=0.25)
                except:
                    st.warning(f"⚠️ Não foi possível calcular KDE para: {cat}")

    if categorias_plotadas == 0:
        st.warning("⚠️ Nenhuma categoria com dados suficientes (mínimo 5 observações)")
    else:
        ax.set_xlabel(var_x, fontsize=11)
        ax.set_ylabel('Frequência aproximada' if not normalizar else 'Densidade', fontsize=11)
        ax.legend(title=hue, fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with st.expander("📊 Ver contagem de observações por categoria"):
        counts = df_filt[hue].value_counts().sort_index()
        st.dataframe(counts.to_frame('Contagem'))

# ===================== ANÁLISES CRUZADAS =====================

st.header("🔀 Análises cruzadas")

col1, col2 = st.columns(2)

with col1:
    if 'Status' in df_filt.columns and 'Score SERASA' in df_filt.columns:
        st.subheader("Score por Status")
        fig1, ax1 = plt.subplots(figsize=(6, 4))

        for s in df_filt["Status"].dropna().unique():
            serie = df_filt.loc[df_filt["Status"] == s, "Score SERASA"].dropna()
            if len(serie) > 5:
                if normalizar:
                    sns.kdeplot(serie, label=f"{s} (n={len(serie)})", 
                               fill=True, alpha=0.25, ax=ax1)
                else:
                    kde = stats.gaussian_kde(serie)
                    x_range = np.linspace(serie.min(), serie.max(), 200)
                    density = kde(x_range)
                    frequency = density * len(serie)
                    ax1.plot(x_range, frequency, label=f"{s} (n={len(serie)})")
                    ax1.fill_between(x_range, frequency, alpha=0.25)

        ax1.set_xlabel("Score SERASA")
        ax1.set_ylabel('Frequência' if not normalizar else 'Densidade')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

with col2:
    if 'Perfil_Risco' in df_filt.columns and 'Idade' in df_filt.columns:
        st.subheader("Idade por Perfil de Risco")
        fig2, ax2 = plt.subplots(figsize=(6, 4))

        for p in df_filt["Perfil_Risco"].dropna().unique():
            serie = df_filt.loc[df_filt["Perfil_Risco"] == p, "Idade"].dropna()
            if len(serie) > 5:
                if normalizar:
                    sns.kdeplot(serie, label=f"{p} (n={len(serie)})", 
                               fill=True, alpha=0.25, ax=ax2)
                else:
                    kde = stats.gaussian_kde(serie)
                    x_range = np.linspace(serie.min(), serie.max(), 200)
                    density = kde(x_range)
                    frequency = density * len(serie)
                    ax2.plot(x_range, frequency, label=f"{p} (n={len(serie)})")
                    ax2.fill_between(x_range, frequency, alpha=0.25)

        ax2.set_xlabel("Idade")
        ax2.set_ylabel('Frequência' if not normalizar else 'Densidade')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

# ===================== TABELA =====================

with st.expander("📋 Ver dados filtrados (primeiros 200 registros)"):
    colunas_display = []
    for col in ["Data", "Estado", "Regiao", "Score SERASA", "Faixa Score", 
                "Idade", "Faixa Idade", "Renda Pres", "Valor Financiado", 
                "Status", "Perfil_Risco"]:
        if col in df_filt.columns:
            colunas_display.append(col)

    st.dataframe(df_filt[colunas_display].head(200))
