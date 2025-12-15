
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuração da página
st.set_page_config(page_title="Dashboard de Análise de Crédito", layout="wide")

# Título
st.title("📊 Dashboard Interativo - Análise de Portfólio de Crédito")

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('Dashboard_Credito_BI.csv', sep=';')

    # Mapear regiões
    regioes = {
        'Norte': ['AC','AP','AM','PA','RO','RR','TO'],
        'Nordeste': ['AL','BA','CE','MA','PB','PE','PI','RN','SE'],
        'Centro-Oeste': ['DF','GO','MT','MS'],
        'Sudeste': ['ES','MG','RJ','SP'],
        'Sul': ['PR','RS','SC']
    }
    def map_regiao(estado):
        for k,v in regioes.items():
            if estado in v: return k
        return 'Outros'
    df['Regiao'] = df['Estado'].apply(map_regiao)

    return df

df = load_data()

# Sidebar com filtros
st.sidebar.header("🔍 Filtros")

# Filtro de Status
status_options = ['Todos'] + list(df['Status'].unique())
status_filter = st.sidebar.multiselect(
    'Status',
    options=status_options,
    default=['Todos']
)

# Filtro de Região
regiao_options = ['Todas'] + list(df['Regiao'].unique())
regiao_filter = st.sidebar.multiselect(
    'Região',
    options=regiao_options,
    default=['Todas']
)

# Filtro de Faixa de Score
score_filter = st.sidebar.multiselect(
    'Faixa de Score',
    options=['Todas'] + list(df['Faixa Score'].dropna().unique()),
    default=['Todas']
)

# Filtro de Faixa de Idade
idade_filter = st.sidebar.multiselect(
    'Faixa de Idade',
    options=['Todas'] + list(df['Faixa Idade'].dropna().unique()),
    default=['Todas']
)

# Aplicar filtros
df_filtered = df.copy()

if 'Todos' not in status_filter and status_filter:
    df_filtered = df_filtered[df_filtered['Status'].isin(status_filter)]

if 'Todas' not in regiao_filter and regiao_filter:
    df_filtered = df_filtered[df_filtered['Regiao'].isin(regiao_filter)]

if 'Todas' not in score_filter and score_filter:
    df_filtered = df_filtered[df_filtered['Faixa Score'].isin(score_filter)]

if 'Todas' not in idade_filter and idade_filter:
    df_filtered = df_filtered[df_filtered['Faixa Idade'].isin(idade_filter)]

# KPIs
st.header("📈 Indicadores Principais")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Propostas", f"{len(df_filtered):,}")
with col2:
    taxa_aprov = (df_filtered['Aprovado_Flag'].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    st.metric("Taxa de Aprovação", f"{taxa_aprov:.1f}%")
with col3:
    score_medio = df_filtered['Score SERASA'].mean() if len(df_filtered) > 0 else 0
    st.metric("Score Médio", f"{score_medio:.0f}")
with col4:
    valor_total = df_filtered['Valor Financiado'].sum() if len(df_filtered) > 0 else 0
    st.metric("Valor Total", f"R$ {valor_total/1e6:.1f}M")

# Seletor de visualizações KDE
st.header("📉 Análises de Distribuição (KDE)")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Variável Principal")
    var_principal = st.selectbox(
        "Selecione a variável para análise:",
        ['Score SERASA', 'Idade', 'Limite de Crédito', 'Renda Pres', 'Valor Financiado']
    )

with col_right:
    st.subheader("Segmentar por")
    segmentar_por = st.selectbox(
        "Selecione a variável de segmentação:",
        ['Status', 'Regiao', 'Faixa Score', 'Faixa Idade', 'Perfil_Risco']
    )

# Gráfico KDE
st.subheader(f"Distribuição de {var_principal} por {segmentar_por}")

fig, ax = plt.subplots(figsize=(12, 6))

# Plotar KDE para cada categoria
for categoria in df_filtered[segmentar_por].dropna().unique():
    dados = df_filtered[df_filtered[segmentar_por] == categoria][var_principal].dropna()
    if len(dados) > 5:
        sns.kdeplot(dados, label=str(categoria), alpha=0.3, ax=ax)

ax.set_xlabel(var_principal, fontsize=12)
ax.set_ylabel('Densidade', fontsize=12)
ax.legend(title=segmentar_por)
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# Análises cruzadas
st.header("🔀 Análises Cruzadas")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Score por Status")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for status in df_filtered['Status'].unique():
        dados = df_filtered[df_filtered['Status'] == status]['Score SERASA'].dropna()
        if len(dados) > 5:
            sns.kdeplot(dados, label=status, fill=True, alpha=0.3, ax=ax1)
    ax1.set_xlabel('Score SERASA')
    ax1.set_ylabel('Densidade')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col2:
    st.subheader("Idade por Perfil de Risco")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for perfil in df_filtered['Perfil_Risco'].dropna().unique():
        dados = df_filtered[df_filtered['Perfil_Risco'] == perfil]['Idade'].dropna()
        if len(dados) > 5:
            sns.kdeplot(dados, label=perfil, fill=True, alpha=0.3, ax=ax2)
    ax2.set_xlabel('Idade')
    ax2.set_ylabel('Densidade')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# Tabela de dados filtrados
with st.expander("📋 Ver Dados Filtrados"):
    st.dataframe(df_filtered[['Data', 'Estado', 'Score SERASA', 'Idade', 'Renda Pres', 
                                'Valor Financiado', 'Status', 'Perfil_Risco']].head(100))

st.sidebar.markdown("---")
st.sidebar.info(f"**Dados filtrados:** {len(df_filtered)} de {len(df)} registros")
