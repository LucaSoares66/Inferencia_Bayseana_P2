import pandas as pd
from sqlalchemy import create_engine, text, inspect
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import pymc as pm
import arviz as az
import patsy
import os
import seaborn as sns
import streamlit as st
import bambi as bmb
from bambi import Prior
import scipy.stats as stats
import plotly.graph_objects as go
import arviz as az
import matplotlib.pyplot as plt
from bambi import Prior
import matplotlib.lines as mlines
import math


# --- Inputs de conexão ---
user = "data_iesb"
password = "iesb"
host = "bigdata.dataiesb.com"
port = 5432
database = "iesb"
schema = "public"

# -------------------------------------------------------------
# 1. CARREGAMENTO DO BANCO DE DADOS ENEM 2024
# -------------------------------------------------------------


user = "data_iesb"
password = "iesb"
host = "bigdata.dataiesb.com"
port = 5432
database = "iesb"
schema = "public"



st.subheader("Problema de pesquisa")
st.write("O quanto a diferença entre ensino público e privado refletem nas notas do vestibular ENEM?")
@st.cache_data(show_spinner=True)
def load_data():
    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    )

    with engine.connect() as conn:
        conn.execute(text(f"SET search_path TO {schema};"))
        conn.commit()

        df_enem = pd.read_sql_table(
            table_name="ed_enem_2024_resultados_amos_per",
            con=engine,
            schema=schema
        )
    return df_enem


with st.spinner("Conectando ao banco e carregando base ENEM 2024..."):
    try:
        df_enem = load_data()
        st.session_state["df_enem"] = df_enem  # 🚀 Guarda no session_state
        st.success("Base carregada com sucesso!")
    except Exception as e:
        st.error("Erro ao carregar a base.")
        st.exception(e)



# -------------------------------------------------------------
# 2. PREPARAÇÃO DA BASE COMPLETA
# -------------------------------------------------------------

if "df_enem" in st.session_state:

    df_enem = st.session_state["df_enem"]

    @st.cache_data(show_spinner=True)
    def preparar_base(df):

        cols_necessarias = [
            "tp_dependencia_adm_esc",
            "nota_cn_ciencias_da_natureza",
            "nota_ch_ciencias_humanas",
            "nota_lc_linguagens_e_codigos",
            "nota_mt_matematica",
            "nota_redacao"
        ]

        df_full = df[cols_necessarias].copy()

        # Limpeza
        df_full['tp_dependencia_adm_esc'] = df_full['tp_dependencia_adm_esc'].str.strip()
        df_full = df_full[df_full['tp_dependencia_adm_esc'] != 'Não Respondeu']
        df_full = df_full.dropna()

        # Variável binária
        df_full['escola_privada'] = np.where(
            df_full['tp_dependencia_adm_esc'] == 'Privada', 1, 0
        )

        return df_full

    with st.spinner("Limpando base, removendo nulos e criando variáveis..."):
        df_full = preparar_base(df_enem)
        st.session_state["df_full"] = df_full   # 🚀 também guarda df_full


else:
    st.info("⚠️ A base ainda não foi carregada.")













# -------------------------------------------------------------
# 1. DEFINIÇÃO DAS PRIORS MANUAIS
# -------------------------------------------------------------
st.subheader("Priors do Modelo")

with st.expander("Priors definidas manualmente", expanded=True):
    st.write("""
    Como estamos usando notas de 0 a 1000:
    
    Para Nota de redação:
    - A prior do coeficiente (beta) tem média de 67.5 (diferença esperada entre privada e pública)  
    - A prior do intercepto tem média 600 (média esperada de escola pública)  
    
    Para demais Notas:
    - A prior do coeficiente (beta) tem média de 60 (diferença esperada entre privada e pública)  
    - A prior do intercepto tem média 500 (média esperada de escola pública) 

    """)





# -------------------------------------------------------------
# 2. CARREGAMENTO DOS MODELOS PRÉ-TREINADOS (.nc)
# -------------------------------------------------------------

# 1. DEFINIÇÃO EXPLÍCITA DAS COLUNAS (Baseado nos seus arquivos da imagem)
# Isso garante que a variável 'cols_notas' sempre exista
cols_notas = [
    'nota_cn_ciencias_da_natureza',
    'nota_ch_ciencias_humanas',
    'nota_lc_linguagens_e_codigos',
    'nota_mt_matematica',
    'nota_redacao'
]

# Salva na sessão para garantir consistência futura
st.session_state["cols_notas"] = cols_notas

# Garante que temos lugar para guardar os resultados na memória
if "idatas" not in st.session_state:
    st.session_state["idatas"] = {}

# 2. FUNÇÃO CACHEADA DE LEITURA
@st.cache_resource(show_spinner=False)
def carregar_inference_data(caminho_arquivo):
    """
    Carrega um arquivo NetCDF (.nc) contendo o InferenceData do Arviz.
    """
    if not os.path.exists(caminho_arquivo):
        return None
    try:
        idata = az.from_netcdf(caminho_arquivo)
        return idata
    except Exception as e:
        return None

# 3. EXECUÇÃO DO LOOP DE LEITURA
colunas_carregadas = []
colunas_erro = []

# Barra de progresso visual
progresso = st.progress(0)
total_cols = len(cols_notas)

for i, col in enumerate(cols_notas):
    # Nome do arquivo: nota_mt_matematica.nc
    nome_arquivo = f"{col}.nc"
    
    # Verifica se já carregamos antes (Session State)
    if col not in st.session_state["idatas"]:
        
        # Tenta carregar do disco
        idata = carregar_inference_data(nome_arquivo)
        
        if idata is not None:
            st.session_state["idatas"][col] = idata
            colunas_carregadas.append(col)
        else:
            colunas_erro.append(f"Arquivo não encontrado ou corrompido: {nome_arquivo}")
            
    else:
        # Já estava na memória RAM
        colunas_carregadas.append(col)
    
    # Atualiza barra
    progresso.progress((i + 1) / total_cols)

progresso.empty()


# -------------------------------------------------------------
# CONFIGURAÇÃO DOS ARQUIVOS E PRIORS
# -------------------------------------------------------------
st.subheader("📊 Análise Visual: Expectativa (Prior) vs. Realidade (Posterior)")

# 1. Defina onde estão os arquivos .nc (Baseado na sua imagem)
# Se estiverem na mesma pasta do script, deixe vazio. Se estiverem em 'resultados_nc', mude aqui.
PASTA_ARQUIVOS = ""  # Ex: "resultados_nc/" ou "" se estiver na raiz

cols_notas = [
    'nota_cn_ciencias_da_natureza',
    'nota_ch_ciencias_humanas',
    'nota_lc_linguagens_e_codigos',
    'nota_mt_matematica',
    'nota_redacao'
]

# 2. Dicionário Visual: Isso diz ao Streamlit onde desenhar a linha pontilhada azul
# (Deve bater com o que você configurou no treinamento)
dict_priors_visual = {
    'nota_cn_ciencias_da_natureza': {'mu': 60,   'sigma': 1.5},
    'nota_ch_ciencias_humanas':     {'mu': 60,   'sigma': 1.5}, # Ajuste conforme seu treino
    'nota_lc_linguagens_e_codigos': {'mu': 60,   'sigma': 1.5},
    'nota_mt_matematica':           {'mu': 60,   'sigma': 1.5},
    'nota_redacao':                 {'mu': 67.5, 'sigma': 1.5}  # Prior Específica
}

# --- Função Auxiliar de KDE (Matemática para suavizar as curvas) ---
def calcular_kde(dados):
    kde = stats.gaussian_kde(dados)
    min_val, max_val = dados.min(), dados.max()
    padding = (max_val - min_val) * 0.2
    x = np.linspace(min_val - padding, max_val + padding, 200)
    y = kde(x)
    return x, y

# -------------------------------------------------------------
# LOOP DE LEITURA E PLOTAGEM
# -------------------------------------------------------------
for col in cols_notas:
    
    # Monta o caminho do arquivo: "nota_redacao.nc"
    caminho_arquivo = os.path.join(PASTA_ARQUIVOS, f"{col}.nc")
    nome_materia = col.replace('nota_', '').replace('_',' ').title()

    # Verifica se o arquivo existe
    if not os.path.exists(caminho_arquivo):
        st.warning(f"Arquivo não encontrado: {caminho_arquivo}")
        continue

    try:
        # A. LEITURA DO ARQUIVO .NC (POSTERIOR / REALIDADE)
        # O Arviz carrega aquele arquivo que você mostrou no print
        idata = az.from_netcdf(caminho_arquivo)
        
        # Extrai a coluna 'escola_privada' (que é o Beta/Impacto)
        # .values.flatten() transforma a matriz complexa numa lista simples de números
        dados_posterior = idata.posterior["escola_privada"].values.flatten()

        # B. GERAÇÃO DA PRIOR (EXPECTATIVA / TEORIA)
        # Busca a configuração no dicionário
        if col in dict_priors_visual:
            params = dict_priors_visual[col]
            mu_plot = params['mu']
            sigma_plot = params['sigma']
            legenda_prior = f"Prior (Expectativa: {mu_plot})"
        else:
            # Fallback
            mu_plot = 40; sigma_plot = 5
            legenda_prior = "Prior (Genérica)"

        # Gera números aleatórios baseados na prior para desenhar a curva
        dados_prior = np.random.normal(loc=mu_plot, scale=sigma_plot, size=10000)

        # C. CÁLCULO DAS CURVAS
        x_prior, y_prior = calcular_kde(dados_prior)
        x_post, y_post = calcular_kde(dados_posterior)

        # D. CONSTRUÇÃO DO GRÁFICO PLOTLY
        fig = go.Figure()

        # Linha Azul Pontilhada (O que a gente achava)
        fig.add_trace(go.Scatter(
            x=x_prior, y=y_prior,
            mode='lines',
            name=legenda_prior,
            line=dict(color='rgba(79, 163, 255, 0.6)', width=2, dash='dash'),
            fill='tozeroy',
            fillcolor='rgba(79, 163, 255, 0.1)'
        ))

        # Linha Vermelha/Laranja Sólida (O que os dados mostraram)
        fig.add_trace(go.Scatter(
            x=x_post, y=y_post,
            mode='lines',
            name='Posterior (Dados Reais)',
            line=dict(color='rgba(255, 79, 109, 0.9)', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 79, 109, 0.2)'
        ))

        # Layout Limpo
        fig.update_layout(
            title=f"<b>{nome_materia}</b>",
            xaxis_title="Diferença de Pontos (Privada - Pública)",
            yaxis_title="Densidade de Probabilidade",
            template="plotly_white",
            height=380,
            hovermode="x unified",
            legend=dict(
                yanchor="top", y=0.98, xanchor="right", x=0.98,
                bgcolor="rgba(255,255,255,0.7)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        
        

    except Exception as e:
        st.error(f"Erro ao processar {col}: {e}")
    
# -------------------------------------------------------------
# 5. TABELA DE INTERVALOS DE CREDIBILIDADE E R-HAT (Via Arquivos)
# -------------------------------------------------------------
st.write("## 📊 Resumo dos Resultados (95% HDI)")
st.markdown("""
Esta tabela resume o **impacto estimado** de estudar em escola privada, lendo diretamente dos modelos salvos:
*   **Efeito Médio:** A diferença média de pontos estimada pelo modelo.
*   **Limites (2.5% - 97.5%):** O intervalo onde temos 95% de certeza que está o valor real.
*   **R-hat:** Diagnóstico de qualidade. Valores **abaixo de 1.05** indicam que o modelo funcionou bem.
""")

# Configuração da lista de matérias (caso não esteja na sessão)
cols_notas = [
    'nota_cn_ciencias_da_natureza',
    'nota_ch_ciencias_humanas',
    'nota_lc_linguagens_e_codigos',
    'nota_mt_matematica',
    'nota_redacao'
]

# Configuração do caminho (deixe vazio "" se estiver na mesma pasta do script)
PASTA_ARQUIVOS = "" 

lista_resultados = []

# --- LOOP DE LEITURA E PROCESSAMENTO ---
for col in cols_notas:
    
    # 1. Monta o caminho do arquivo
    caminho_arquivo = os.path.join(PASTA_ARQUIVOS, f"{col}.nc")
    
    # 2. Verifica se existe
    if not os.path.exists(caminho_arquivo):
        # Se não achar o arquivo, pula silenciosamente ou avisa (opcional)
        continue

    try:
        # 3. Carrega o arquivo do disco
        idata_atual = az.from_netcdf(caminho_arquivo)

        # 4. Gera o sumário estatístico (A mágica do Arviz)
        # var_names=["escola_privada"] foca apenas no coeficiente que nos interessa
        summary = az.summary(
            idata_atual,
            var_names=["escola_privada"],
            hdi_prob=0.95
        )

        # 5. Extrai os valores do DataFrame do summary
        # O summary retorna um DF onde o index é o nome da variável ('escola_privada')
        media = summary['mean'].values[0]
        hdi_lower = summary['hdi_2.5%'].values[0]
        hdi_upper = summary['hdi_97.5%'].values[0]
        r_hat = summary['r_hat'].values[0]

        lista_resultados.append({
            'Disciplina': col.replace('nota_', '').replace('_', ' ').title(),
            'Efeito Médio': media,
            'Limite Inferior': hdi_lower,
            'Limite Superior': hdi_upper,
            'R-hat': r_hat
        })
        
    except Exception as e:
        st.error(f"Erro ao ler arquivo {col}.nc: {e}")

# --- MONTAGEM DA TABELA ---

if lista_resultados:
    # Criação do DataFrame
    df_credibilidade = pd.DataFrame(lista_resultados)

    # Define a disciplina como índice
    df_credibilidade.set_index('Disciplina', inplace=True)

    # Função para pintar o R-hat
    def style_rhat(v):
        if v > 1.05:
            return 'color: red; font-weight: bold;'
        return 'color: green;'

    # Aplica estilos e exibe
    st.dataframe(
        df_credibilidade.style
        .background_gradient(cmap='Blues', subset=['Efeito Médio'])
        .map(style_rhat, subset=['R-hat'])
        .format("{:.2f}"), 
        use_container_width=True
    )
else:
    st.warning("Nenhum arquivo de modelo (.nc) foi encontrado para gerar a tabela.")

# -------------------------------------------------------------
# 6. MÉTRICAS DE PREVISÃO (Via Arquivos .nc)
# -------------------------------------------------------------
st.write("## 📉 Performance Preditiva")
st.markdown("""
Avaliamos o quanto o modelo consegue explicar a variação das notas lendo os modelos salvos.
*   **R²:** Quanto da nota é explicado apenas pelo fato de ser escola Privada/Pública.
*   **RMSE/MAE:** O erro médio do modelo (em pontos).
""")



df_full = st.session_state["df_full"]

# Lista de matérias
cols_notas = [
    'nota_cn_ciencias_da_natureza',
    'nota_ch_ciencias_humanas',
    'nota_lc_linguagens_e_codigos',
    'nota_mt_matematica',
    'nota_redacao'
]

# Configuração do caminho (vazio se estiver na raiz)
PASTA_ARQUIVOS = ""

metrics_list = []

# --- LOOP DE CÁLCULO ---
for col in cols_notas:
    
    # 1. Monta caminho e verifica existência
    caminho_arquivo = os.path.join(PASTA_ARQUIVOS, f"{col}.nc")
    
    if not os.path.exists(caminho_arquivo):
        continue # Pula se não tiver o arquivo
    
    try:
        # 2. Carrega o modelo salvo
        idata = az.from_netcdf(caminho_arquivo)
        
        # 3. Extrai os Coeficientes Médios (Intercepto e Beta)
        # O .item() converte de array numpy 0-d para um float simples
        intercept_mean = idata.posterior['Intercept'].mean().values
        beta_mean = idata.posterior['escola_privada'].mean().values
        
        # 4. Dados Reais (Ground Truth)
        y_true = df_full[col].values
        x_escola = df_full['escola_privada'].values
        
        # 5. Cálculo da Previsão (Vetorizado = Instantâneo)
        # Fórmula: Y = Alpha + Beta * X
        y_pred_mean = intercept_mean + (beta_mean * x_escola)
        
        # 6. Cálculo das Métricas (Numpy puro)
        
        # RMSE (Raiz do Erro Quadrático Médio)
        rmse = np.sqrt(np.mean((y_true - y_pred_mean) ** 2))
        
        # MAE (Erro Absoluto Médio)
        mae = np.mean(np.abs(y_true - y_pred_mean))
        
        # R2 (Coeficiente de Determinação)
        ss_res = np.sum((y_true - y_pred_mean) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        metrics_list.append({
            'Disciplina': col.replace('nota_', '').replace('_', ' ').title(),
            'R² (Explicação)': r2,
            'RMSE (Erro Pontos)': rmse,
            'MAE (Erro Pontos)': mae
        })
        
    except Exception as e:
        st.warning(f"Não foi possível calcular métricas para {col}: {e}")

# --- TABELA FINAL ---
if metrics_list:
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.set_index('Disciplina', inplace=True)

    # Estilização
    st.dataframe(
        df_metrics.style
        .background_gradient(cmap='Greens', subset=['R² (Explicação)']) # Quanto maior melhor
        .background_gradient(cmap='Reds', subset=['RMSE (Erro Pontos)'])  # Quanto maior pior
        .format("{:.3f}", subset=['R² (Explicação)'])
        .format("{:.1f}", subset=['RMSE (Erro Pontos)', 'MAE (Erro Pontos)']),
        use_container_width=True
    )

    st.caption("Nota: O R² baixo é esperado. O desempenho do aluno depende de muitos outros fatores (renda, estudo, família) além do tipo de escola.")
else:
    st.warning("Nenhum modelo encontrado para calcular métricas.")

# -------------------------------------------------------------
# 6. CHECAGEM VISUAL (PPC) - ESTILO STREAMLIT
# -------------------------------------------------------------
st.write("## 👁️ Checagem Visual do Modelo (PPC)")
st.markdown("""
Comparamos a distribuição das **notas reais** (linha vermelha) com o que o modelo **imagina** 
que as notas deveriam ser (linhas azuis). Se as linhas azuis cobrirem bem a linha preta, 
o modelo entendeu bem os dados.
""")

# Recupera dados
if "idatas" not in st.session_state or "df_full" not in st.session_state:
    st.stop()

idatas = st.session_state["idatas"]
df_enem = st.session_state["df_full"]
cols_notas = st.session_state["cols_notas"]

# Função auxiliar de densidade (KDE)
def get_kde(data):
    # Remove NaNs se houver e calcula densidade
    data = data[~np.isnan(data)]
    if len(data) == 0: return np.array([]), np.array([])
    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 100)
    return x, kde(x)

# --- CONFIGURAÇÃO DO GRID (2 Colunas) ---
cols_screen = st.columns(2)

for idx, col_name in enumerate(cols_notas):
    
    # Define em qual coluna da tela vai plotar (zig-zag)
    col_screen = cols_screen[idx % 2]
    
    with col_screen:
        if col_name not in idatas:
            continue
            
        idata = idatas[col_name]
        nome_titulo = col_name.replace('nota_', '').replace('_', ' ').title()
        
        # 1. DADOS REAIS
        y_real = df_enem[col_name].values
        x_real_plot, y_real_plot = get_kde(y_real)
        
        # 2. SIMULAÇÃO RÁPIDA (FAST PPC)
        # Recupera os parâmetros médios aprendidos pelo modelo
        try:
            alpha = idata.posterior["Intercept"].mean().values
            beta = idata.posterior["escola_privada"].mean().values
            # O Bambi geralmente chama o erro de 'sigma' ou 'nome_da_coluna_sigma'
            # Vamos tentar pegar o sigma genérico
            if "sigma" in idata.posterior:
                sigma = idata.posterior["sigma"].mean().values
            else:
                # Fallback: estimativa rápida do desvio padrão residual
                sigma = np.std(y_real) 
            
            # Variável explicativa
            X = df_enem["escola_privada"].values
            
            # Gera 30 simulações do dataset inteiro (Rápido com Numpy!)
            # Y_sim = (Alpha + Beta*X) + Erro
            mu = alpha + (beta * X)
            
        except Exception as e:
            st.error(f"Erro ao ler parâmetros de {nome_titulo}")
            continue

        # 3. CONSTRUÇÃO DO GRÁFICO PLOTLY
        fig = go.Figure()

        # A. Plota as 30 linhas de simulação (linhas finas azuis)
        for i in range(30):
            # Gera ruído aleatório para cada simulação
            noise = np.random.normal(0, sigma, size=len(X))
            y_sim = mu + noise
            
            x_sim_plot, dens_sim_plot = get_kde(y_sim)
            
            fig.add_trace(go.Scatter(
                x=x_sim_plot, y=dens_sim_plot,
                mode='lines',
                line=dict(color='rgba(0, 122, 204, 0.15)', width=1), # Azul transparente
                hoverinfo='skip', # Não mostrar tooltip nessas linhas para não poluir
                showlegend=False
            ))

        # Adiciona uma linha "fantasma" azul só para aparecer na legenda corretamente
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name='Simulação do Modelo',
            line=dict(color='rgb(0, 122, 204)', width=2)
        ))

        # B. Plota os Dados Reais (Linha Preta grossa)
        fig.add_trace(go.Scatter(
            x=x_real_plot, y=y_real_plot,
            mode='lines',
            name='Dados Observados (Reais)',
            line=dict(color='red', width=2.5)
        ))

        # Layout Limpo
        fig.update_layout(
            title=f"<b>{nome_titulo}</b>",
            xaxis_title="Nota (Escala 0-1000)",
            yaxis_showticklabels=False, # Esconde eixo Y (densidade é abstrata)
            yaxis_title="",
            template="plotly_white",
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(
                yanchor="top", y=0.99, xanchor="right", x=0.99,
                bgcolor="rgba(255,255,255,0.5)"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)