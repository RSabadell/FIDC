
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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

# ===================== RIDGE PLOT COM FACETGRID (JOYPLOT) =====================

st.header("🏔️ Ridge Plot - Estilo JoyPlot (FacetGrid)")

st.info("💡 Gráfico de densidade sobreposta no estilo clássico. Estados ordenados por média.")

if 'Estado' in df_filt.columns and 'Score SERASA' in df_filt.columns:

    # Configurações
    col_ridge1, col_ridge2, col_ridge3, col_ridge4 = st.columns(4)

    with col_ridge1:
        min_obs_ridge = st.slider(
            "Mínimo de observações",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
        )

    with col_ridge2:
        var_ridge = st.selectbox(
            "Variável",
            ["Score SERASA", "Idade", "Valor Financiado"],
        )

    with col_ridge3:
        paleta_ridge = st.selectbox(
            "Paleta de cores",
            ["cubehelix", "rocket", "mako", "viridis", "RdYlGn", "coolwarm"],
        )

    with col_ridge4:
        bw_adjust_val = st.slider(
            "Suavização (bw_adjust)",
            min_value=0.3,
            max_value=2.0,
            value=0.8,
            step=0.1,
        )

    # Preparar dados
    df_ridge = df_filt[['Estado', var_ridge]].dropna().copy()

    # Filtrar estados com amostra mínima
    estados_contagem = df_ridge.groupby('Estado')[var_ridge].count()
    estados_validos = estados_contagem[estados_contagem >= min_obs_ridge].index
    df_ridge = df_ridge[df_ridge['Estado'].isin(estados_validos)]

    if len(estados_validos) < 3:
        st.warning(f"⚠️ Apenas {len(estados_validos)} estado(s) disponível(is). Reduza o mínimo de observações.")
    else:
        # Calcular média por estado e ordenar
        score_medio = df_ridge.groupby('Estado')[var_ridge].mean().sort_values()
        ordem_estados = score_medio.index.tolist()
        n_estados = len(ordem_estados)

        # Criar coluna ordenada como categoria
        df_ridge['Estado_ord'] = pd.Categorical(
            df_ridge['Estado'],
            categories=ordem_estados,
            ordered=True
        )

        # Renomear variável para facilitar
        df_ridge['valor'] = df_ridge[var_ridge]

        # Configurar tema seaborn
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Criar paleta
        if paleta_ridge == "cubehelix":
            pal = sns.cubehelix_palette(n_estados, rot=-.25, light=.7)
        elif paleta_ridge == "RdYlGn":
            pal = sns.color_palette("RdYlGn", n_estados)
        else:
            pal = sns.color_palette(paleta_ridge, n_estados)

        # Criar FacetGrid
        g = sns.FacetGrid(
            df_ridge, 
            row="Estado_ord", 
            hue="Estado_ord", 
            aspect=15, 
            height=0.5, 
            palette=pal
        )

        # Desenhar as densidades (preenchida)
        g.map(sns.kdeplot, "valor",
              bw_adjust=bw_adjust_val, 
              clip_on=False,
              fill=True, 
              alpha=1, 
              linewidth=1)

        # Desenhar linha branca por cima
        g.map(sns.kdeplot, "valor", 
              clip_on=False, 
              color="b", 
              lw=.5, 
              bw_adjust=bw_adjust_val)

        # Linha de referência em y=0
        g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

        # Função para labels com média
        def label(x, color, label):
            ax = plt.gca()
            # Calcular média do estado atual
            estado_atual = label.iloc[0] if hasattr(label, 'iloc') else str(label)
            media = df_ridge[df_ridge['Estado_ord'] == estado_atual]['valor'].mean()

            # Label com estado e média
            texto = f"{estado_atual}  ({media:.0f})"
            ax.text(0, .2, texto, 
                   fontweight="bold", 
                   color=color,
                   ha="left", 
                   va="center", 
                   transform=ax.transAxes,
                   fontsize=8)

        g.map(label, "valor")

        # Ajustar sobreposição
        g.figure.subplots_adjust(hspace=-.25)

        # Limpar eixos
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        # Adicionar título e label do eixo X
        g.figure.suptitle(f'Distribuição de {var_ridge} por Estado', 
                         fontsize=16, fontweight='bold', y=0.98)
        g.set_axis_labels(var_ridge, "")

        # Ajustar layout
        plt.tight_layout()

        # Mostrar no streamlit
        st.pyplot(g.figure)

        # Resetar tema
        sns.set_theme()

        # Estatísticas
        with st.expander("📊 Estatísticas por Estado"):
            stats_estado = df_ridge.groupby('Estado')['valor'].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            stats_estado.columns = ['Contagem', 'Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo']
            stats_estado = stats_estado.sort_values('Média', ascending=False)
            st.dataframe(stats_estado, use_container_width=True)

else:
    st.warning("⚠️ Colunas necessárias não encontradas para Ridge Plot")

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
            key='kde_var'
        )

    with cr:
        hue = st.selectbox(
            "Segmentar por (hue)",
            segmentacoes_disponiveis,
            key='kde_hue'
        )

    col_norm1, col_norm2 = st.columns([2, 1])
    with col_norm1:
        normalizar = st.checkbox(
            "Normalizar curvas (cada curva com área = 1)", 
            value=False,
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
                    data=serie,
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
                    st.warning(f"⚠️ Erro ao calcular KDE para: {cat}")

    if categorias_plotadas == 0:
        st.warning("⚠️ Nenhuma categoria com dados suficientes")
    else:
        ax.set_xlabel(var_x, fontsize=11)
        ax.set_ylabel('Frequência aproximada' if not normalizar else 'Densidade', fontsize=11)
        ax.legend(title=hue, fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with st.expander("📊 Ver contagem por categoria"):
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
                    sns.kdeplot(data=serie, label=f"{s} (n={len(serie)})", 
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
                    sns.kdeplot(data=serie, label=f"{p} (n={len(serie)})", 
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
