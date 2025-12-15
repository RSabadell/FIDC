
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuração básica da página
st.set_page_config(
    page_title="Dashboard de Crédito",
    layout="wide",
)

@st.cache_data
def load_data():
    df = pd.read_csv("Dashboard_Credito_BI.csv", sep=";")
    return df

df = load_data()

st.title("📊 Dashboard Interativo - Portfólio de Crédito")

# ===================== SIDEBAR – FILTROS =====================

st.sidebar.header("🔍 Filtros")

status_opts = sorted(df["Status"].dropna().unique())
status_sel = st.sidebar.multiselect(
    "Status",
    options=status_opts,
    default=status_opts,
)

regiao_opts = sorted(df["Regiao"].dropna().unique())
regiao_sel = st.sidebar.multiselect(
    "Região",
    options=regiao_opts,
    default=regiao_opts,
)

faixa_score_opts = sorted(df["Faixa Score"].dropna().unique())
faixa_score_sel = st.sidebar.multiselect(
    "Faixa de Score",
    options=faixa_score_opts,
    default=faixa_score_opts,
)

faixa_idade_opts = sorted(df["Faixa Idade"].dropna().unique())
faixa_idade_sel = st.sidebar.multiselect(
    "Faixa de Idade",
    options=faixa_idade_opts,
    default=faixa_idade_opts,
)

# aplica filtros
df_filt = df.copy()
if status_sel:
    df_filt = df_filt[df_filt["Status"].isin(status_sel)]
if regiao_sel:
    df_filt = df_filt[df_filt["Regiao"].isin(regiao_sel)]
if faixa_score_sel:
    df_filt = df_filt[df_filt["Faixa Score"].isin(faixa_score_sel)]
if faixa_idade_sel:
    df_filt = df_filt[df_filt["Faixa Idade"].isin(faixa_idade_sel)]

st.sidebar.markdown("---")
st.sidebar.write(f"Registros filtrados: **{len(df_filt)}**")

# ===================== KPIs =====================

st.header("📈 Indicadores principais")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total de Propostas", f"{len(df_filt):,}".replace(",", "."))
with c2:
    taxa_aprov = (
        df_filt["Aprovado_Flag"].sum() / len(df_filt) * 100
        if len(df_filt) > 0 else 0
    )
    st.metric("Taxa de Aprovação", f"{taxa_aprov:.1f}%")
with c3:
    score_med = df_filt["Score SERASA"].mean() if len(df_filt) > 0 else 0
    st.metric("Score Médio", f"{score_med:.0f}")
with c4:
    valor_tot = df_filt["Valor Financiado"].sum() if len(df_filt) > 0 else 0
    st.metric("Valor Total", f"R$ {valor_tot:,.0f}".replace(",", "."))

# ===================== KDE CONFIGURÁVEL - SEM NORMALIZAÇÃO =====================

st.header("📉 Distribuições (KDE) - Frequências Reais")

st.info("💡 **Importante**: Estes gráficos mantêm as proporções reais entre categorias. "
        "Categorias com mais observações terão curvas maiores.")

cl, cr = st.columns(2)

with cl:
    var_x = st.selectbox(
        "Variável contínua para o eixo X",
        ["Score SERASA", "Idade", "Limite de Crédito", "Renda Pres", "Valor Financiado"],
    )

with cr:
    hue = st.selectbox(
        "Segmentar por (hue)",
        ["Status", "Regiao", "Faixa Score", "Faixa Idade", "Perfil_Risco"],
    )

# Opção de normalização
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

# Plotar KDE para cada categoria
for cat in df_filt[hue].dropna().unique():
    serie = df_filt.loc[df_filt[hue] == cat, var_x].dropna()

    if len(serie) > 5:
        if normalizar:
            # Modo padrão: normalizado (densidade)
            sns.kdeplot(
                serie,
                label=f"{cat} (n={len(serie)})",
                fill=True,
                alpha=0.25,
                linewidth=1.5,
                ax=ax,
            )
        else:
            # Modo SEM normalização: escalado pelo número de observações
            # Usar common_norm=False e multiplicar pela contagem
            from scipy import stats

            # Calcular KDE
            kde = stats.gaussian_kde(serie)
            x_range = np.linspace(serie.min(), serie.max(), 200)
            density = kde(x_range)

            # ESCALAR pela contagem para mostrar frequências reais
            frequency = density * len(serie)

            ax.plot(x_range, frequency, label=f"{cat} (n={len(serie)})", linewidth=1.5)
            ax.fill_between(x_range, frequency, alpha=0.25)

ax.set_xlabel(var_x, fontsize=11)

if normalizar:
    ax.set_ylabel('Densidade (normalizada)', fontsize=11)
else:
    ax.set_ylabel('Frequência aproximada', fontsize=11)

ax.legend(title=hue, fontsize=9)
ax.grid(alpha=0.3)

st.pyplot(fig)

# Mostrar contagens por categoria
with st.expander("📊 Ver contagem de observações por categoria"):
    counts = df_filt[hue].value_counts().sort_index()
    st.dataframe(counts.to_frame('Contagem'))

# ===================== ANÁLISES CRUZADAS RÁPIDAS =====================

st.header("🔀 Análises cruzadas")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Score por Status")
    fig1, ax1 = plt.subplots(figsize=(6, 4))

    for s in df_filt["Status"].dropna().unique():
        serie = df_filt.loc[df_filt["Status"] == s, "Score SERASA"].dropna()
        if len(serie) > 5:
            if normalizar:
                sns.kdeplot(serie, label=f"{s} (n={len(serie)})", 
                           fill=True, alpha=0.25, ax=ax1)
            else:
                from scipy import stats
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
    st.subheader("Idade por Perfil de Risco")
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    for p in df_filt["Perfil_Risco"].dropna().unique():
        serie = df_filt.loc[df_filt["Perfil_Risco"] == p, "Idade"].dropna()
        if len(serie) > 5:
            if normalizar:
                sns.kdeplot(serie, label=f"{p} (n={len(serie)})", 
                           fill=True, alpha=0.25, ax=ax2)
            else:
                from scipy import stats
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

with st.expander("📋 Ver dados filtrados (amostra)"):
    st.dataframe(
        df_filt[
            [
                "Data",
                "Estado",
                "Regiao",
                "Score SERASA",
                "Faixa Score",
                "Idade",
                "Faixa Idade",
                "Renda Pres",
                "Valor Financiado",
                "Status",
                "Perfil_Risco",
            ]
        ].head(200)
    )
