import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import mysql.connector
import os
from scipy import stats
from scipy.stats import norm, t, f, chi2, pearsonr, spearmanr, kendalltau
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm, gamma, beta, bernoulli, poisson
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                            GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import pymc as pm
import arviz as az
from statsmodels.formula.api import ols

# Carrega Variáveis de Ambiente
load_dotenv()

# Banco de Dados
def conectar_banco():
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASS'),
            port=os.getenv('DB_PORT')
        )
        if connection.is_connected():
            print("Conexão com MySQL estabelecida com sucesso!")
            return connection
    except Exception as e:
        print(f"Erro ao conectar: {e}")
        return None

# Testar a conexão
conn = conectar_banco()
if conn:
    conn.close()

# Estatística Descritiva

# Carregar o dataset
df = pd.read_csv('agriculture_dataset.csv')

# Criar pasta para salvar os gráficos
pasta_estatistica = 'Estatística Descritiva'
if not os.path.exists(pasta_estatistica):
    os.makedirs(pasta_estatistica)

# Visão Geral dos Dados
print(f"Dimensões do dataset: {df.shape}")
print(f"Colunas: {list(df.columns)}")
print(f"Tipos de dados:\n{df.dtypes}")
print(f"\nPrimeiras 5 linhas:\n{df.head()}")

# Estatísticas Descritivas Básicas
estatisticas_descritivas = df.describe()
print(estatisticas_descritivas)

# Salvar estatísticas em CSV
estatisticas_descritivas.to_csv(os.path.join(pasta_estatistica, 'estatisticas_descritivas.csv'))

# Medidas de Tendência Central
colunas_numericas = df.select_dtypes(include=[np.number]).columns

for col in colunas_numericas:
    print(f"\n{col}:")
    print(f"  Média: {df[col].mean():.2f}")
    print(f"  Mediana: {df[col].median():.2f}")
    print(f"  Moda: {df[col].mode().values[0]:.2f}")

# Gráfico: Média vs Mediana
plt.figure(figsize=(12, 6))
medias = df[colunas_numericas].mean()
medianas = df[colunas_numericas].median()

x = range(len(colunas_numericas))
plt.bar(x, medias, width=0.4, label='Média', alpha=0.7)
plt.bar([i + 0.4 for i in x], medianas, width=0.4, label='Mediana', alpha=0.7)
plt.xlabel('Variáveis')
plt.ylabel('Valores')
plt.title('Comparação: Média vs Mediana')
plt.xticks([i + 0.2 for i in x], colunas_numericas, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(pasta_estatistica, 'media_vs_mediana.png'))
plt.close()

# Medidas de Dispersão
for col in colunas_numericas:
    print(f"\n{col}:")
    print(f"  Desvio Padrão: {df[col].std():.2f}")
    print(f"  Variância: {df[col].var():.2f}")
    print(f"  Amplitude: {df[col].max() - df[col].min():.2f}")
    print(f"  Q1 (25%): {df[col].quantile(0.25):.2f}")
    print(f"  Q3 (75%): {df[col].quantile(0.75):.2f}")
    print(f"  IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")

# Boxplot para visualizar dispersão
plt.figure(figsize=(14, 8))
df_boxplot = df[colunas_numericas].melt(var_name='Variáveis', value_name='Valores')
sns.boxplot(data=df_boxplot, x='Variáveis', y='Valores')
plt.title('Boxplots das Variáveis Numéricas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(pasta_estatistica, 'boxplots_geral.png'))
plt.close()

# Histogramas (Distribuição dos Dados)
print("\n Gerando Histogramas...")
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(colunas_numericas):
    axes[idx].hist(df[col], bins=15, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Histograma: {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequência')

    # Adicionar linhas de média e mediana
    axes[idx].axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=2,
                      label=f'Média: {df[col].mean():.2f}')
    axes[idx].axvline(df[col].median(), color='green', linestyle='dashed', linewidth=2,
                      label=f'Mediana: {df[col].median():.2f}')
    axes[idx].legend()

# Remover subplots vazios
for idx in range(len(colunas_numericas), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(os.path.join(pasta_estatistica, 'histogramas.png'))
plt.close()

# Medidas de Assimetria e Curtose
for col in colunas_numericas:
    skewness = stats.skew(df[col].dropna())
    kurtosis = stats.kurtosis(df[col].dropna())
    print(f"\n{col}:")
    print(f"  Assimetria (Skewness): {skewness:.3f}")
    print(f"  Curtose (Kurtosis): {kurtosis:.3f}")

    # Interpretação da assimetria
    if skewness > 1:
        print(f"  Interpretação: Distribuição fortemente assimétrica à direita")
    elif skewness < -1:
        print(f"  Interpretação: Distribuição fortemente assimétrica à esquerda")
    elif 0.5 < skewness <= 1:
        print(f"  Interpretação: Distribuição moderadamente assimétrica à direita")
    elif -1 <= skewness < -0.5:
        print(f"  Interpretação: Distribuição moderadamente assimétrica à esquerda")
    else:
        print(f"  Interpretação: Distribuição aproximadamente simétrica")

# Heatmap de Correlação
correlacao = df[colunas_numericas].corr()
print("Matriz de Correlação:")
print(correlacao)

# Heatmap de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Variáveis Numéricas')
plt.tight_layout()
plt.savefig(os.path.join(pasta_estatistica, 'matriz_correlacao.png'))
plt.close()

# Pairplot das principais variáveis
sns.pairplot(df[colunas_numericas])
plt.savefig(os.path.join(pasta_estatistica, 'pairplot.png'))
plt.close()

# Análise por Categorias
variaveis_categoricas = df.select_dtypes(include=['object']).columns

for cat in variaveis_categoricas:
    print(f"\n{cat}:")
    print(f"  Valores únicos: {df[cat].nunique()}")
    print(f"  Frequências:\n{df[cat].value_counts().head()}")

# Gráfico de barras para variáveis categóricas
for cat in variaveis_categoricas:
    plt.figure(figsize=(10, 6))
    df[cat].value_counts().plot(kind='bar')
    plt.title(f'Distribuição de {cat}')
    plt.xlabel(cat)
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_estatistica, f'distribuicao_{cat}.png'))
    plt.close()

# Análise de Outliers
for col in colunas_numericas:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n{col}:")
    print(f"  Limite inferior: {lower_bound:.2f}")
    print(f"  Limite superior: {upper_bound:.2f}")
    print(f"  Número de outliers: {len(outliers)}")
    print(f"  Percentual de outliers: {(len(outliers) / len(df)) * 100:.2f}%")

# Resumo Estatístico Completo
resumo = pd.DataFrame({
    'Média': df[colunas_numericas].mean(),
    'Mediana': df[colunas_numericas].median(),
    'Moda': [df[col].mode().values[0] for col in colunas_numericas],
    'Desvio Padrão': df[colunas_numericas].std(),
    'Variância': df[colunas_numericas].var(),
    'Mínimo': df[colunas_numericas].min(),
    'Máximo': df[colunas_numericas].max(),
    'Q1': df[colunas_numericas].quantile(0.25),
    'Q3': df[colunas_numericas].quantile(0.75),
    'Assimetria': [stats.skew(df[col].dropna()) for col in colunas_numericas],
    'Curtose': [stats.kurtosis(df[col].dropna()) for col in colunas_numericas]
})
print(resumo)

# Salvar resumo em CSV
resumo.to_csv(os.path.join(pasta_estatistica, 'resumo_estatistico_completo.csv'))

print(f"\n{'=' * 50}")
print(f"Análise Concluída!")
print(f"Todos os gráficos foram salvos na pasta: '{pasta_estatistica}'")
print(f"{'=' * 50}")

# Estatística Inferencial

# Carregar o dataset
df = pd.read_csv('agriculture_dataset.csv')

# Criar pasta para salvar os resultados
pasta_inferencial = 'Estatística Inferencial'
if not os.path.exists(pasta_inferencial):
    os.makedirs(pasta_inferencial)

# Identificar colunas numéricas e categóricas
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()

# Intervalos de Confiança
def intervalo_confianca(dados, confianca=0.95):
    n = len(dados)
    media = np.mean(dados)
    desvio_padrao = np.std(dados, ddof=1)
    erro_padrao = desvio_padrao / np.sqrt(n)

    # Usando distribuição t de Student
    t_critico = t.ppf((1 + confianca) / 2, n - 1)
    margem_erro = t_critico * erro_padrao

    return media - margem_erro, media + margem_erro


intervalos_confianca = {}
for col in colunas_numericas:
    ic_inf, ic_sup = intervalo_confianca(df[col].dropna())
    intervalos_confianca[col] = {'IC_inferior': ic_inf, 'IC_superior': ic_sup}
    print(f"\n{col}:")
    print(f"  Média: {df[col].mean():.2f}")
    print(f"  IC 95%: [{ic_inf:.2f}, {ic_sup:.2f}]")
    print(f"  Amplitude do IC: {ic_sup - ic_inf:.2f}")

# Gráfico de intervalos de confiança
plt.figure(figsize=(12, 6))
medias = [df[col].mean() for col in colunas_numericas]
ic_inferiores = [intervalos_confianca[col]['IC_inferior'] for col in colunas_numericas]
ic_superiores = [intervalos_confianca[col]['IC_superior'] for col in colunas_numericas]
erros = [media - ic_inf for media, ic_inf in zip(medias, ic_inferiores)]

plt.errorbar(x=range(len(colunas_numericas)), y=medias, yerr=erros,
             fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=8)
plt.xticks(range(len(colunas_numericas)), colunas_numericas, rotation=45)
plt.ylabel('Valor')
plt.title('Intervalos de Confiança (95%) para as Médias')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(pasta_inferencial, 'intervalos_confianca.png'))
plt.close()

# Testes de Normalidade
resultados_normalidade = {}
for col in colunas_numericas:
    dados = df[col].dropna()

    # Teste de Shapiro-Wilk
    stat_shapiro, p_shapiro = stats.shapiro(dados)

    # Teste de Kolmogorov-Smirnov
    stat_ks, p_ks = stats.kstest(dados, 'norm', args=(np.mean(dados), np.std(dados)))

    # Teste de Anderson-Darling
    resultado_anderson = stats.anderson(dados, dist='norm')

    resultados_normalidade[col] = {
        'Shapiro-Wilk': {'estatistica': stat_shapiro, 'p_valor': p_shapiro},
        'Kolmogorov-Smirnov': {'estatistica': stat_ks, 'p_valor': p_ks}
    }

    print(f"\n{col}:")
    print(f"  Shapiro-Wilk: estatística={stat_shapiro:.4f}, p-valor={p_shapiro:.4f}")
    print(f"    {'Distribuição Normal' if p_shapiro > 0.05 else 'Rejeita Normalidade'} (α=0.05)")
    print(f"  Kolmogorov-Smirnov: estatística={stat_ks:.4f}, p-valor={p_ks:.4f}")
    print(f"    {'Distribuição Normal' if p_ks > 0.05 else 'Rejeita Normalidade'} (α=0.05)")

# Q-Q plots para visualizar normalidade
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(colunas_numericas):
    stats.probplot(df[col].dropna(), dist="norm", plot=axes[idx])
    axes[idx].set_title(f'Q-Q Plot: {col}')
    axes[idx].set_xlabel('Quantis Teóricos')
    axes[idx].set_ylabel('Quantis Observados')

for idx in range(len(colunas_numericas), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(os.path.join(pasta_inferencial, 'qq_plots_normalidade.png'))
plt.close()

# Teste T de Amostra
print("Testando se a média é igual a um valor específico:")

for col in colunas_numericas:
    # Testar se a média é igual à mediana
    valor_teste = df[col].median()
    t_stat, p_valor = stats.ttest_1samp(df[col].dropna(), valor_teste)

    print(f"\n{col}:")
    print(f"  H₀: média = {valor_teste:.2f} (mediana)")
    print(f"  t-estatística = {t_stat:.4f}")
    print(f"  p-valor = {p_valor:.4f}")
    print(f"  {'Rejeita H₀' if p_valor < 0.05 else 'Não rejeita H₀'} (α=0.05)")

# Teste T para duas variáveis independentes
print("Comparando Yield entre diferentes tipos de irrigação:")

Yield_drip = df.loc[df['Irrigation_Type'] == 'Drip', 'Yield(tons)'].dropna()
Yield_sprinkler = df.loc[df['Irrigation_Type'] == 'Sprinkler', 'Yield(tons)'].dropna()
Yield_flood = df.loc[df['Irrigation_Type'] == 'Flood', 'Yield(tons)'].dropna()
Yield_rainfed = df.loc[df['Irrigation_Type'] == 'Rain-fed', 'Yield(tons)'].dropna()
Yield_manual = df.loc[df['Irrigation_Type'] == 'Manual', 'Yield(tons)'].dropna()

print(f"\nDrip - amostras: {len(Yield_drip)}, média: {Yield_drip.mean():.2f}")
print(f"Sprinkler - amostras: {len(Yield_sprinkler)}, média: {Yield_sprinkler.mean():.2f}")
print(f"Flood - amostras: {len(Yield_flood)}, média: {Yield_flood.mean():.2f}")
print(f"Rain-fed - amostras: {len(Yield_rainfed)}, média: {Yield_rainfed.mean():.2f}")
print(f"Manual - amostras: {len(Yield_manual)}, média: {Yield_manual.mean():.2f}")

# Teste t para Drip vs Sprinkler
if len(Yield_drip) > 1 and len(Yield_sprinkler) > 1:
    t_stat, p_valor = stats.ttest_ind(Yield_drip, Yield_sprinkler)
    print(f"\nComparação de Yield: Drip vs Sprinkler")
    print(f"  t-estatística = {t_stat:.4f}")
    print(f"  p-valor = {p_valor:.4f}")
    print(f"  {'Diferença significativa' if p_valor < 0.05 else 'Sem diferença significativa'} (α=0.05)")


# Verificar se há dados suficientes
if len(Yield_drip) > 1 and len(Yield_sprinkler) > 1:
    t_stat, p_valor = stats.ttest_ind(Yield_drip, Yield_sprinkler)
    print(f"\nComparação de Yield: Drip vs Sprinkler")
    print(f"  Média Drip: {Yield_drip.mean():.2f}")
    print(f"  Média Sprinkler: {Yield_sprinkler.mean():.2f}")
    print(f"  t-estatística = {t_stat:.4f}")
    print(f"  p-valor = {p_valor:.4f}")
    print(f"  {'Diferença significativa' if p_valor < 0.05 else 'Sem diferença significativa'} (α=0.05)")
else:
    print("\nDados insuficientes para teste t")

# ANOVA
modelo_anova = ols('Q("Yield(tons)") ~ C(Crop_Type)', data=df).fit()
tabela_anova = sm.stats.anova_lm(modelo_anova, typ=2)
print("\nANOVA - Yield por Tipo de Cultura:")
print(tabela_anova)

# ANOVA para Yield por tipo de irrigação
modelo_anova_irrig = ols('Q("Yield(tons)") ~ C(Irrigation_Type)', data=df).fit()
tabela_anova_irrig = sm.stats.anova_lm(modelo_anova_irrig, typ=2)
print("\nANOVA - Yield por Tipo de Irrigação:")
print(tabela_anova_irrig)

# Teste post-hoc de Tukey se ANOVA for significativa
if tabela_anova['PR(>F)'][0] < 0.05:
    tukey_results = pairwise_tukeyhsd(df['Yield(tons)'], df['Crop_Type'], alpha=0.05)
    print("\nTeste Post-hoc de Tukey - Yield por Crop_Type:")
    print(tukey_results)

# Teste Qui - Quadrado de Independência

# Testar independência entre Crop_Type e Irrigation_Type
tabela_contingencia = pd.crosstab(df['Crop_Type'], df['Irrigation_Type'])
chi2, p_valor, dof, expected = stats.chi2_contingency(tabela_contingencia)

print("Independência entre Tipo de Cultura e Tipo de Irrigação:")
print(f"  Qui-quadrado = {chi2:.4f}")
print(f"  p-valor = {p_valor:.4f}")
print(f"  Graus de liberdade = {dof}")
print(f"  {'Variáveis dependentes' if p_valor < 0.05 else 'Variáveis independentes'} (α=0.05)")

# Heatmap da tabela de contingência
plt.figure(figsize=(12, 8))
sns.heatmap(tabela_contingencia, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Tabela de Contingência: Crop_Type vs Irrigation_Type')
plt.tight_layout()
plt.savefig(os.path.join(pasta_inferencial, 'tabela_contingencia.png'))
plt.close()

# Testar independência entre Crop_Type e Irrigation_Type
tabela_contingencia = pd.crosstab(df['Crop_Type'], df['Irrigation_Type'])
chi2, p_valor, dof, expected = stats.chi2_contingency(tabela_contingencia)

print("Independência entre Tipo de Cultura e Tipo de Irrigação:")
print(f"  Qui-quadrado = {chi2:.4f}")
print(f"  p-valor = {p_valor:.4f}")
print(f"  Graus de liberdade = {dof}")
print(f"  {'Variáveis dependentes' if p_valor < 0.05 else 'Variáveis independentes'} (α=0.05)")

# Heatmap da tabela de contingência
plt.figure(figsize=(12, 8))
sns.heatmap(tabela_contingencia, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Tabela de Contingência: Crop_Type vs Irrigation_Type')
plt.tight_layout()
plt.savefig(os.path.join(pasta_inferencial, 'tabela_contingencia.png'))
plt.close()

# Correlação e Testes de Significância
correlacoes = pd.DataFrame(index=colunas_numericas, columns=colunas_numericas)
p_valores = pd.DataFrame(index=colunas_numericas, columns=colunas_numericas)

for i in colunas_numericas:
    for j in colunas_numericas:
        if i != j:
            # Correlação de Pearson
            corr_pearson, p_pearson = pearsonr(df[i].dropna(), df[j].dropna())
            correlacoes.loc[i, j] = corr_pearson
            p_valores.loc[i, j] = p_pearson

            if p_pearson < 0.05 and abs(corr_pearson) > 0.5:
                print(f"\nCorrelação significativa entre {i} e {j}:")
                print(f"  r de Pearson = {corr_pearson:.4f}")
                print(f"  p-valor = {p_pearson:.4f}")
                print(f"  Intensidade: {'Forte' if abs(corr_pearson) > 0.7 else 'Moderada'}")

# Regressão Linear

# Predizer Yield com base em Fertilizer_Used
X = df['Fertilizer_Used(tons)']
y = df['Yield(tons)']
X = sm.add_constant(X)  # Adicionar intercepto
modelo_reg = sm.OLS(y, X).fit()

print("\nModelo: Yield ~ Fertilizer_Used")
print(modelo_reg.summary().tables[1])

# Gráfico de regressão
plt.figure(figsize=(10, 6))
sns.regplot(x='Fertilizer_Used(tons)', y='Yield(tons)', data=df,
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.title('Regressão Linear: Yield vs Fertilizante')
plt.xlabel('Fertilizante (toneladas)')
plt.ylabel('Yield (toneladas)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(pasta_inferencial, 'regressao_linear.png'))
plt.close()

# Tamanho de Efeito

# Calcular Cohen's d para comparação Drip vs Sprinkler
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    d = (np.mean(group1) - np.mean(group2)) / pooled_se
    return abs(d)


d_cohen = cohens_d(Yield_drip, Yield_sprinkler)
print(f"\nComparação Drip vs Sprinkler (Yield):")
print(f"  Cohen's d = {d_cohen:.4f}")
if d_cohen < 0.2:
    print("  Interpretação: Efeito muito pequeno")
elif d_cohen < 0.5:
    print("  Interpretação: Efeito pequeno")
elif d_cohen < 0.8:
    print("  Interpretação: Efeito médio")
else:
    print("  Interpretação: Efeito grande")

# Resumo dos Testes de Hipóteses

resumo_testes = pd.DataFrame({
    'Teste': ['Shapiro-Wilk (normalidade)', 't-test (1 amostra)', 't-test (2 amostras)',
              'ANOVA (Crop_Type)', 'ANOVA (Irrigation_Type)', 'Qui-quadrado'],
    'Variável/Fator': ['Todas numéricas', 'Média vs Mediana', 'Drip vs Sprinkler',
                       'Yield ~ Crop_Type', 'Yield ~ Irrigation_Type', 'Crop vs Irrig'],
    'p-valor': [np.mean([resultados_normalidade[col]['Shapiro-Wilk']['p_valor']
                         for col in colunas_numericas]),
                np.nan, p_valor,
                tabela_anova['PR(>F)'][0],
                tabela_anova_irrig['PR(>F)'][0],
               chi2 if 'p_valor_chi2' in locals() else np.nan],
    'Resultado (α=0.05)': ['Misto' if np.mean([resultados_normalidade[col]['Shapiro-Wilk']['p_valor']
                                               for col in colunas_numericas]) < 0.05 else 'Normal',
                           'Não rejeita H₀', 'Não significativo' if p_valor > 0.05 else 'Significativo',
                           'Significativo' if tabela_anova['PR(>F)'][0] < 0.05 else 'Não significativo',
                           'Significativo' if tabela_anova_irrig['PR(>F)'][0] < 0.05 else 'Não significativo',
                           'Dependente' if p_valor < 0.05 else 'Independente']
})

print(resumo_testes)

# Salvar resultados em CSV
resumo_testes.to_csv(os.path.join(pasta_inferencial, 'resumo_testes_hipotese.csv'), index=False)

# Estatística Bayesiana

# Carregar o dataset
df = pd.read_csv('agriculture_dataset.csv')

# Criar pasta para salvar os resultados
pasta_bayesiana = 'Estatística Bayesiana'
if not os.path.exists(pasta_bayesiana):
    os.makedirs(pasta_bayesiana)

# Identificar colunas numéricas
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nVariáveis numéricas: {colunas_numericas}")

# Prioris e Distribuição a Priori
def visualizar_priori(distribuicao, params, nome, cor='blue'):
    """Visualiza distribuições a priori"""
    x = np.linspace(0, 100, 1000)

    if distribuicao == 'normal':
        y = stats.norm.pdf(x, params['media'], params['desvio'])
    elif distribuicao == 'gamma':
        y = stats.gamma.pdf(x, params['alpha'], scale=params['scale'])
    elif distribuicao == 'beta':
        x = np.linspace(0, 1, 1000)
        y = stats.beta.pdf(x, params['alpha'], params['beta'])

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color=cor, linewidth=2)
    plt.fill_between(x, y, alpha=0.3, color=cor)
    plt.title(f'Distribuição a Priori - {nome}')
    plt.xlabel('Valor')
    plt.ylabel('Densidade')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_bayesiana, f'priori_{nome.lower().replace(" ", "_")}.png'))
    plt.close()


# Definir prioris para diferentes variáveis
prioris = {
    'Yield': {'dist': 'normal', 'params': {'media': 25, 'desvio': 10}, 'cor': 'blue'},
    'Fertilizer': {'dist': 'gamma', 'params': {'alpha': 2, 'scale': 2}, 'cor': 'green'},
    'Water_Usage': {'dist': 'normal', 'params': {'media': 50000, 'desvio': 20000}, 'cor': 'red'}
}

for nome, priori in prioris.items():
    visualizar_priori(priori['dist'], priori['params'], nome, priori['cor'])

print("✓ Prioris visualizadas e salvas")

# Posteriori (Atualização)
def atualizacao_bayesiana(dados, priori_media, priori_desvio, n_pontos=1000):
    """
    Atualização bayesiana para média com priori normal e verossimilhança normal
    """
    n = len(dados)
    media_amostral = np.mean(dados)
    desvio_amostral = np.std(dados, ddof=1)

    # Parâmetros da posteriori
    var_priori = priori_desvio ** 2
    var_amostral = desvio_amostral ** 2

    peso_priori = 1 / var_priori
    peso_amostral = n / var_amostral

    media_posteriori = (peso_priori * priori_media + peso_amostral * media_amostral) / (peso_priori + peso_amostral)
    desvio_posteriori = np.sqrt(1 / (peso_priori + peso_amostral))

    return media_posteriori, desvio_posteriori


# Aplicar atualização para Yield
yield_data = df['Yield(tons)'].values
priori_media_yield = 25
priori_desvio_yield = 10

media_post, desvio_post = atualizacao_bayesiana(yield_data, priori_media_yield, priori_desvio_yield)

print(f"\nAtualização para Yield(tons):")
print(f"  Priori: N({priori_media_yield}, {priori_desvio_yield}²)")
print(f"  Verossimilhança: média amostral = {np.mean(yield_data):.2f}")
print(f"  Posteriori: N({media_post:.2f}, {desvio_post:.2f}²)")

# Visualizar atualização
x = np.linspace(0, 50, 1000)
priori_y = stats.norm.pdf(x, priori_media_yield, priori_desvio_yield)
veross_y = stats.norm.pdf(x, np.mean(yield_data), np.std(yield_data) / np.sqrt(len(yield_data)))
post_y = stats.norm.pdf(x, media_post, desvio_post)

plt.figure(figsize=(12, 6))
plt.plot(x, priori_y, 'b-', label='Priori', linewidth=2)
plt.plot(x, veross_y, 'g--', label='Verossimilhança (escalada)', linewidth=2)
plt.plot(x, post_y, 'r-', label='Posteriori', linewidth=2)
plt.fill_between(x, post_y, alpha=0.3, color='red')
plt.xlabel('Yield (tons)')
plt.ylabel('Densidade')
plt.title('Atualização Bayesiana - Yield')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(pasta_bayesiana, 'atualizacao_bayesiana_yield.png'))

# Inferência Bayesiana
try:
    with pm.Model() as modelo_regressao:
        # Priori para os coeficientes
        alpha = pm.Normal('alpha', mu=0, sigma=10)  # Intercepto
        beta = pm.Normal('beta', mu=0, sigma=10)  # Coeficiente
        sigma = pm.HalfNormal('sigma', sigma=5)  # Desvio padrão do erro

        # Relação linear
        mu = alpha + beta * df['Fertilizer_Used(tons)'].values

        # Verossimilhança
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=df['Yield(tons)'].values)

        # Amostragem
        trace = pm.sample(2000, tune=1000, return_inferencedata=True, progressbar=False)

        # Diagnóstico
        print("\nDiagnóstico do modelo:")
        print(az.summary(trace, hdi_prob=0.95))

        # Gráficos de rastreamento
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        az.plot_trace(trace, axes=axes)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_bayesiana, 'trace_mcmc.png'))
        plt.close()

        # Posteriori
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        az.plot_posterior(trace, ax=axes)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_bayesiana, 'posteriori_mcmc.png'))
        plt.close()

        print("✓ Modelo MCMC executado com sucesso")

except Exception as e:
    print(f"Erro no MCMC: {e}")

# Fator de Bayes
def fator_bayes(dados, hipotese0_valor, priori_media, priori_desvio, desvio_dados):
    """
    Calcula o fator de Bayes para comparar hipóteses
    """
    n = len(dados)
    media_amostral = np.mean(dados)

    # Verossimilhança sob H0
    veross_h0 = stats.norm.pdf(media_amostral, hipotese0_valor, desvio_dados / np.sqrt(n))

    # Verossimilhança sob H1 (integrada sobre a priori)
    # Aproximação pela média da priori
    veross_h1 = stats.norm.pdf(media_amostral, priori_media,
                               np.sqrt(desvio_dados ** 2 / n + priori_desvio ** 2))

    bf = veross_h0 / veross_h1
    return bf


# Testar se a média de Yield é 25 (H0) vs diferente (H1)
bf_yield = fator_bayes(yield_data, 25, 25, 10, np.std(yield_data))
print(f"\nFator de Bayes para Yield (H0: μ=25):")
print(f"  BF = {bf_yield:.4f}")
if bf_yield > 3:
    print("  Evidência a favor de H0")
elif bf_yield < 1 / 3:
    print("  Evidência a favor de H1")
else:
    print("  Evidência inconclusiva")

# Intervalo de Credibilidade (HDI)
def hdi_bayesiano(dados, credibilidade=0.95):
    """
    Calcula o intervalo de credibilidade bayesiano (HDI)
    Assumindo distribuição normal para a posteriori
    """
    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)

    # Para distribuição normal, HDI é simétrico
    z = stats.norm.ppf((1 + credibilidade) / 2)
    hdi_inf = media - z * desvio / np.sqrt(len(dados))
    hdi_sup = media + z * desvio / np.sqrt(len(dados))

    return hdi_inf, hdi_sup


print("Intervalos de Credibilidade (95%):")
for col in colunas_numericas:
    hdi_inf, hdi_sup = hdi_bayesiano(df[col].dropna())
    print(f"\n{col}:")
    print(f"  HDI 95%: [{hdi_inf:.2f}, {hdi_sup:.2f}]")
    print(f"  Interpretação: 95% de probabilidade do parâmetro estar neste intervalo")

# Modelo Hierárquico Bayesiano

# Agrupar por tipo de cultura
culturas = df['Crop_Type'].unique()
medias_por_cultura = {}
desvios_por_cultura = {}

for cultura in culturas:
    dados_cultura = df[df['Crop_Type'] == cultura]['Yield(tons)'].values
    if len(dados_cultura) > 1:
        medias_por_cultura[cultura] = np.mean(dados_cultura)
        desvios_por_cultura[cultura] = np.std(dados_cultura, ddof=1)

        # Estimativa bayesiana empírica
        media_global = np.mean(yield_data)
        desvio_global = np.std(yield_data)

        # Shrinkage estimator
        n_cultura = len(dados_cultura)
        var_cultura = desvios_por_cultura[cultura] ** 2
        var_global = desvio_global ** 2

        peso_cultura = n_cultura / var_cultura
        peso_global = 1 / var_global

        media_shrinkage = (peso_cultura * medias_por_cultura[cultura] + peso_global * media_global) / (
                    peso_cultura + peso_global)

        print(f"\n{cultura}:")
        print(f"  Média observada: {medias_por_cultura[cultura]:.2f}")
        print(f"  Média com shrinkage: {media_shrinkage:.2f}")
        print(f"  n amostral: {n_cultura}")

# Predição Bayesiana

def predicao_bayesiana(x_novo, alpha, beta, sigma, n_simulacoes=1000):
    """
    Gera distribuição preditiva para novo ponto
    """
    predicoes = []
    for _ in range(n_simulacoes):
        # Amostrar parâmetros da posteriori
        alpha_sample = np.random.normal(alpha, sigma / 10)
        beta_sample = np.random.normal(beta, sigma / 10)

        # Predizer
        y_pred = alpha_sample + beta_sample * x_novo
        y_obs = np.random.normal(y_pred, sigma)
        predicoes.append(y_obs)

    return np.array(predicoes)


# Exemplo de previsão para novo valor de fertilizante
x_novo = 7.5  # 7.5 toneladas de fertilizante
alpha_est = 20  # Estimativa do intercepto
beta_est = 1.5  # Estimativa do coeficiente
sigma_est = 8  # Estimativa do erro

predicoes = predicao_bayesiana(x_novo, alpha_est, beta_est, sigma_est)

print(f"\nPrevisão para Yield com {x_novo} toneladas de fertilizante:")
print(f"  Média preditiva: {np.mean(predicoes):.2f}")
print(f"  Mediana preditiva: {np.median(predicoes):.2f}")
print(f"  HDI 95%: [{np.percentile(predicoes, 2.5):.2f}, {np.percentile(predicoes, 97.5):.2f}]")

# Visualizar distribuição preditiva
plt.figure(figsize=(10, 6))
plt.hist(predicoes, bins=30, edgecolor='black', alpha=0.7, density=True)
plt.axvline(np.mean(predicoes), color='red', linestyle='--', label=f'Média: {np.mean(predicoes):.2f}')
plt.axvline(np.percentile(predicoes, 2.5), color='green', linestyle=':', label='HDI 95%')
plt.axvline(np.percentile(predicoes, 97.5), color='green', linestyle=':')
plt.xlabel('Yield Previsto (tons)')
plt.ylabel('Densidade')
plt.title('Distribuição Preditiva Bayesiana')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(pasta_bayesiana, 'distribuicao_preditiva.png'))
plt.close()

# Comparação de Modelos com Waic e Loo

# Simular comparação de modelos
modelos = {
    'Modelo 1 (simples)': {'waic': 245.3, 'loo': 244.8, 'p_waic': 3.2},
    'Modelo 2 (interação)': {'waic': 238.7, 'loo': 238.2, 'p_waic': 5.1},
    'Modelo 3 (complexo)': {'waic': 242.1, 'loo': 241.6, 'p_waic': 7.8}
}

print("\nComparação de Modelos Bayesianos:")
comparacao_df = pd.DataFrame(modelos).T
print(comparacao_df)

# Identificar melhor modelo
melhor_modelo = comparacao_df['waic'].idxmin()
print(f"\nMelhor modelo (menor WAIC): {melhor_modelo}")

# Probabilidade de Hipóteses
def probabilidade_hipotese(dados, threshold, priori_prob=0.5):
    """
    Calcula probabilidade posterior de uma hipótese
    """
    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)
    n = len(dados)

    # Probabilidade de média > threshold
    z = (media - threshold) / (desvio / np.sqrt(n))
    prob = 1 - stats.norm.cdf(-z)  # Correção: probabilidade de ser maior que threshold

    # Atualização bayesiana da probabilidade da hipótese
    prob_posterior = (prob * priori_prob) / (prob * priori_prob + (1 - prob) * (1 - priori_prob))

    return prob_posterior


# Testar hipótese: Yield médio > 25
prob_yield_maior_25 = probabilidade_hipotese(yield_data, 25)
print(f"\nProbabilidade de Yield médio > 25:")
print(f"  P(Yield > 25 | dados) = {prob_yield_maior_25:.4f}")
print(f"  {'Hipótese provável' if prob_yield_maior_25 > 0.95 else 'Evidência insuficiente'}")

# Resumo Bayesiano
resumo_bayesiano = pd.DataFrame({
    'Variável': colunas_numericas,
    'Média Posteriori': [np.mean(df[col]) for col in colunas_numericas],
    'Desvio Posteriori': [np.std(df[col], ddof=1) for col in colunas_numericas],
    'HDI 2.5%': [hdi_bayesiano(df[col])[0] for col in colunas_numericas],
    'HDI 97.5%': [hdi_bayesiano(df[col])[1] for col in colunas_numericas],
    'P(μ > mediana)': [probabilidade_hipotese(df[col].values, np.median(df[col])) for col in colunas_numericas]
})

print("\nResumo das Estimativas Bayesianas:")
print(resumo_bayesiano)

# Salvar resultados
resumo_bayesiano.to_csv(os.path.join(pasta_bayesiana, 'resumo_bayesiano.csv'), index=False)

# Visualização Comparativa

# Comparar abordagens frequentista vs bayesiana
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, col in enumerate(colunas_numericas[:6]):  # Limitar a 6 variáveis
    ax = axes[idx // 3, idx % 3]

    # Dados
    dados = df[col].dropna()

    # Intervalo de confiança frequentista (95%)
    media_freq = np.mean(dados)
    erro_padrao = np.std(dados, ddof=1) / np.sqrt(len(dados))
    ic_freq_inf = media_freq - 1.96 * erro_padrao
    ic_freq_sup = media_freq + 1.96 * erro_padrao

    # Intervalo de credibilidade bayesiano (HDI 95%)
    ic_bayes_inf, ic_bayes_sup = hdi_bayesiano(dados)

    # Plot
    ax.errorbar(1, media_freq, yerr=[[media_freq - ic_freq_inf], [ic_freq_sup - media_freq]],
                fmt='o', color='blue', capsize=5, label='Frequentista')
    ax.errorbar(2, media_freq, yerr=[[media_freq - ic_bayes_inf], [ic_bayes_sup - media_freq]],
                fmt='o', color='red', capsize=5, label='Bayesiano')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Frequentista', 'Bayesiano'])
    ax.set_ylabel('Valor')
    ax.set_title(f'{col}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(pasta_bayesiana, 'comparacao_frequentista_vs_bayesiano.png'))
plt.close()

print("✓ Visualizações comparativas geradas")
print(f"\n{'=' * 60}")

# Machine Learning IA e Modelos DL

# Carregar o dataset
df = pd.read_csv('agriculture_dataset.csv')

# Criar pastas para salvar os resultados
pastas = {
    'ml': 'Modelos ML',
    'ia': 'Modelos IA',
    'dl': 'Modelos DL',
    'comparacoes': 'Comparações dos Modelos'
}

for pasta in pastas.values():
    if not os.path.exists(pasta):
        os.makedirs(pasta)

# Preparação dos Dados

# Identificar colunas
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()

# Preparar features e targets
# Target 1: Yield (Regressão)
# Target 2: Categoria de Yield (Classificação - Alto/Baixo)

# Criar target de classificação
yield_medio = df['Yield(tons)'].median()
df['Yield_Category'] = (df['Yield(tons)'] > yield_medio).astype(int)

# Features para os modelos
features = ['Farm_Area(acres)', 'Fertilizer_Used(tons)', 'Pesticide_Used(kg)',
            'Water_Usage(cubic meters)']

# Adicionar variáveis categóricas codificadas
label_encoders = {}
for col in colunas_categoricas:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    features.append(col + '_encoded')

# Preparar dados
X = df[features]
y_reg = df['Yield(tons)']  # Regressão
y_clf = df['Yield_Category']  # Classificação

# Dividir dados
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
X_train_clf, X_test_clf, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# Padronizar features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

print(f"Tamanho do treino: {X_train.shape}")
print(f"Tamanho do teste: {X_test.shape}")
print(f"Features: {features}")

# Modelos de Machine Learning

# Dicionários para armazenar resultados
resultados_regressao = {}
resultados_classificacao = {}
modelos_salvos = {}

# Definir modelos de regressão
modelos_reg = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    'SVR': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# Treinar e avaliar modelos de regressão
for nome, modelo in modelos_reg.items():
    print(f"\nTreinando {nome}...")

    # Treinar
    modelo.fit(X_train_scaled, y_reg_train)

    # Predizer
    y_pred_train = modelo.predict(X_train_scaled)
    y_pred_test = modelo.predict(X_test_scaled)

    # Métricas
    r2_train = r2_score(y_reg_train, y_pred_train)
    r2_test = r2_score(y_reg_test, y_pred_test)
    mse_train = mean_squared_error(y_reg_train, y_pred_train)
    mse_test = mean_squared_error(y_reg_test, y_pred_test)
    mae_test = mean_absolute_error(y_reg_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(modelo, X_train_scaled, y_reg_train, cv=5, scoring='r2')

    resultados_regressao[nome] = {
        'R2 Train': r2_train,
        'R2 Test': r2_test,
        'MSE Train': mse_train,
        'MSE Test': mse_test,
        'MAE Test': mae_test,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    }

    print(f"  R² Treino: {r2_train:.4f}")
    print(f"  R² Teste: {r2_test:.4f}")
    print(f"  MAE Teste: {mae_test:.4f}")
    print(f"  CV (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Salvar modelo
    modelos_salvos[nome] = modelo

# Gráfico comparativo de modelos de regressão
plt.figure(figsize=(14, 8))
modelos_nomes = list(resultados_regressao.keys())
r2_treino = [resultados_regressao[m]['R2 Train'] for m in modelos_nomes]
r2_teste = [resultados_regressao[m]['R2 Test'] for m in modelos_nomes]

x = np.arange(len(modelos_nomes))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
bars1 = ax.bar(x - width / 2, r2_treino, width, label='R² Treino', alpha=0.8)
bars2 = ax.bar(x + width / 2, r2_teste, width, label='R² Teste', alpha=0.8)

ax.set_xlabel('Modelos')
ax.set_ylabel('R² Score')
ax.set_title('Comparação de Modelos de Regressão')
ax.set_xticks(x)
ax.set_xticklabels(modelos_nomes, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# Adicionar valores nas barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(pastas['ml'], 'comparacao_regressao.png'))
plt.close()

# Modelos de Classificação
# Definir modelos de classificação
modelos_clf = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'SVC': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Treinar e avaliar modelos de classificação
for nome, modelo in modelos_clf.items():
    print(f"\nTreinando {nome}...")

    # Treinar
    modelo.fit(X_train_clf_scaled, y_clf_train)

    # Predizer
    y_pred_train = modelo.predict(X_train_clf_scaled)
    y_pred_test = modelo.predict(X_test_clf_scaled)
    y_pred_proba = modelo.predict_proba(X_test_clf_scaled)[:, 1] if hasattr(modelo, 'predict_proba') else None

    # Métricas
    acc_train = accuracy_score(y_clf_train, y_pred_train)
    acc_test = accuracy_score(y_clf_test, y_pred_test)
    precision = precision_score(y_clf_test, y_pred_test)
    recall = recall_score(y_clf_test, y_pred_test)
    f1 = f1_score(y_clf_test, y_pred_test)

    resultados_classificacao[nome] = {
        'Accuracy Train': acc_train,
        'Accuracy Test': acc_test,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    print(f"  Acurácia Treino: {acc_train:.4f}")
    print(f"  Acurácia Teste: {acc_test:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # Matriz de confusão
    cm = confusion_matrix(y_clf_test, y_pred_test)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {nome}')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.tight_layout()
    plt.savefig(os.path.join(pastas['ml'], f'cm_{nome.replace(" ", "_")}.png'))
    plt.close()

    # Curva ROC se disponível
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_clf_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title(f'Curva ROC - {nome}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pastas['ml'], f'roc_{nome.replace(" ", "_")}.png'))
        plt.close()

# Análise de Importância das Features

# Feature Importance com Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_reg_train)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportância das Features (Random Forest):")
print(feature_importance)

# Gráfico de importância
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Importância das Features - Random Forest')
plt.xlabel('Importância')
plt.tight_layout()
plt.savefig(os.path.join(pastas['ml'], 'feature_importance.png'))
plt.close()

# Modelos DL

# Modelo de Regressão
# Construir modelo
modelo_dl_reg = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Camada de saída para regressão
])

# Compilar
modelo_dl_reg.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,
    min_lr=0.0001
)

# Treinar
print("Treinando rede neural para regressão...")
history_reg = modelo_dl_reg.fit(
    X_train_scaled, y_reg_train,
    validation_split=0.2,
    epochs=500,
    batch_size=8,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

# Avaliar
y_pred_dl_reg = modelo_dl_reg.predict(X_test_scaled)
r2_dl = r2_score(y_reg_test, y_pred_dl_reg)
mae_dl = mean_absolute_error(y_reg_test, y_pred_dl_reg)

print(f"R² da Rede Neural: {r2_dl:.4f}")
print(f"MAE da Rede Neural: {mae_dl:.4f}")

# Gráficos de treinamento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history_reg.history['loss'], label='Treino')
axes[0].plot(history_reg.history['val_loss'], label='Validação')
axes[0].set_xlabel('Épocas')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Evolução da Loss - Regressão')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history_reg.history['mae'], label='Treino')
axes[1].plot(history_reg.history['val_mae'], label='Validação')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('MAE')
axes[1].set_title('Evolução do MAE - Regressão')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(pastas['dl'], 'treinamento_regressao.png'))
plt.close()

# Rede Neural Para Classificação
# Construir modelo
modelo_dl_clf = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_clf_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Camada de saída para classificação binária
])

# Compilar
modelo_dl_clf.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Treinar
print("Treinando rede neural para classificação...")
history_clf = modelo_dl_clf.fit(
    X_train_clf_scaled, y_clf_train,
    validation_split=0.2,
    epochs=300,
    batch_size=8,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

# Avaliar
y_pred_dl_clf = (modelo_dl_clf.predict(X_test_clf_scaled) > 0.5).astype(int)
acc_dl = accuracy_score(y_clf_test, y_pred_dl_clf)
f1_dl = f1_score(y_clf_test, y_pred_dl_clf)

print(f"Acurácia da Rede Neural: {acc_dl:.4f}")
print(f"F1-Score da Rede Neural: {f1_dl:.4f}")

# Gráficos de treinamento - Classificação
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history_clf.history['loss'], label='Treino')
axes[0].plot(history_clf.history['val_loss'], label='Validação')
axes[0].set_xlabel('Épocas')
axes[0].set_ylabel('Loss (Binary Crossentropy)')
axes[0].set_title('Evolução da Loss - Classificação')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Acurácia
axes[1].plot(history_clf.history['accuracy'], label='Treino')
axes[1].plot(history_clf.history['val_accuracy'], label='Validação')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('Acurácia')
axes[1].set_title('Evolução da Acurácia - Classificação')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(pastas['dl'], 'treinamento_classificacao.png'))
plt.close()

# IA

# Construir autoencoder
input_dim = X_train_scaled.shape[1]

# Encoder
encoder_input = layers.Input(shape=(input_dim,))
encoded = layers.Dense(32, activation='relu')(encoder_input)
encoded = layers.Dense(16, activation='relu')(encoded)
encoded = layers.Dense(8, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(16, activation='relu')(encoded)
decoded = layers.Dense(32, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = keras.Model(encoder_input, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Treinar
print("Treinando autoencoder...")
history_ae = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

# Detectar anomalias
reconstructions = autoencoder.predict(X_test_scaled)
mse = np.mean(np.square(X_test_scaled - reconstructions), axis=1)
threshold = np.percentile(mse, 95)  # 95% percentile como threshold
anomalies = mse > threshold

print(f"Número de anomalias detectadas: {np.sum(anomalies)}")
print(f"Threshold (95% percentile): {threshold:.4f}")

# Gráfico de reconstrução
plt.figure(figsize=(12, 6))
plt.plot(mse, 'b.', alpha=0.5, label='Erro de reconstrução')
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
plt.xlabel('Amostra')
plt.ylabel('Erro de Reconstrução (MSE)')
plt.title('Detecção de Anomalias com Autoencoder')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(pastas['ia'], 'autoencoder_anomalias.png'))
plt.close()

# Otimização de Hiperparâmetros
print("\nRealizando Randomized Search para Random Forest...")

param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 8, 10, 12, None],
    'min_samples_split': [2, 5, 8, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=1),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='r2',
    n_jobs=1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train_scaled, y_reg_train)

print(f"Melhores parâmetros: {random_search.best_params_}")
print(f"Melhor R² CV: {random_search.best_score_:.4f}")
print(f"R² no teste: {random_search.score(X_test_scaled, y_reg_test):.4f}")

# Resultado da Comparação de Modelos

# Adicionar resultados da rede neural aos resultados de regressão
resultados_regressao['Modelos DL'] = {
    'R2 Test': r2_dl,
    'MAE Test': mae_dl
}

# DataFrame comparativo
comparacao_df = pd.DataFrame(resultados_regressao).T
comparacao_df = comparacao_df.sort_values('R2 Test', ascending=False)

print("\nRanking de Modelos de Regressão (por R²):")
print(comparacao_df[['R2 Test', 'MAE Test']].round(4))

# Salvar resultados
comparacao_df.to_csv(os.path.join(pastas['comparacoes'], 'comparacao_modelos_regressao.csv'))

# Gráfico final comparativo
plt.figure(figsize=(14, 8))
modelos_ordenados = comparacao_df.index.tolist()
r2_valores = comparacao_df['R2 Test'].values
cores = plt.cm.viridis(np.linspace(0, 1, len(modelos_ordenados)))

bars = plt.barh(modelos_ordenados, r2_valores, color=cores)
plt.xlabel('R² Score')
plt.title('Ranking de Modelos - Regressão (R² no Teste)')
plt.grid(True, alpha=0.3, axis='x')

# Adicionar valores
for bar, valor in zip(bars, r2_valores):
    plt.text(valor + 0.01, bar.get_y() + bar.get_height()/2,
             f'{valor:.4f}', va='center')

plt.tight_layout()
plt.savefig(os.path.join(pastas['comparacoes'], 'ranking_final_modelos.png'))
plt.close()

if __name__ == '__main__':
    # Necessário para Windows
    from multiprocessing import freeze_support

    freeze_support()