# EXTRAÇÃO E RECONHECIMENTO DOS DADOS
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar os dados
url = "https://raw.githubusercontent.com/alura-cursos/challenge2-data-science/refs/heads/main/TelecomX_Data.json"
df = pd.read_json(url)

# Normalizar as colunas 'customer', 'phone', 'internet', 'account'
df_customer = pd.json_normalize(df['customer'])
df_phone = pd.json_normalize(df['phone'])
df_internet = pd.json_normalize(df['internet'])
df_account = pd.json_normalize(df['account'])

# # Exibir os DataFrames normalizados para inspeção no Google Colab
# print("Customer DataFrame:")
# display(df_customer.head())

# print("\nPhone DataFrame:")
# display(df_phone.head())

# print("\nInternet DataFrame:")
# display(df_internet.head())

# print("\nAccount DataFrame:")
# display(df_account.head())

# Concatenar todos os DataFrames normalizados com o DataFrame original
df_final = pd.concat([df.drop(['customer', 'phone', 'internet', 'account'], axis=1),
                      df_customer, df_phone, df_internet, df_account], axis=1)

# # Exibir o DataFrame final
# print("\nFinal DataFrame:")
# display(df_final.head())

# verificar se algum é null
# list_df = [df_customer, df_phone, df_internet, df_account]
# for i in list_df:
#   print(i.isnull().sum())
#   print("\n")
df_final.head()



### TRANSFORMAÇÕES BÁSICAS


# Renomear colunas para o português
df_final = df_final.rename(columns={
    'customerID': 'ID_Cliente',
    'Churn': 'Evasao',
    'gender': 'Genero',
    'SeniorCitizen': 'Idoso',
    'Partner': 'Possui_Parceiro',
    'Dependents': 'Possui_Dependentes',
    'tenure': 'Meses_Contrato',
    'PhoneService': 'Telefone',
    'MultipleLines': 'Multiplas_Linhas',
    'InternetService': 'Tipo_Internet',
    'OnlineSecurity': 'Seguranca_Online',
    'OnlineBackup': 'Backup_Online',
    'DeviceProtection': 'Protecao_Dispositivo',
    'TechSupport': 'Suporte_Tecnico',
    'StreamingTV': 'Streaming_TV',
    'StreamingMovies': 'Streaming_Filmes',
    'Contract': 'Tipo_Contrato',
    'PaperlessBilling': 'Fatura_Papel',
    'PaymentMethod': 'Metodo_Pagamento',
    'Charges.Monthly': 'Cobranca_Mensal',
    'Charges.Total': 'Cobranca_Total'
})

# Substituir os valores da coluna 'Evasao'
df_final['Evasao'] = df_final['Evasao'].map({
    'Yes': 'Saíram da base',
    'No': 'Continuam na base'
})

# Traduzir os valores da coluna 'Tipo_Contrato' para português
df_final['Tipo_Contrato'] = df_final['Tipo_Contrato'].map({
    'Month-to-month': 'Mensal',
    'One year': 'Anual (1 ano)',
    'Two year': 'Bienal (2 anos)'
})

#Mapeamento das nomes para
mapeamento_nomes = {
    'Cobranca_Mensal': 'Cobrança Mensal (R$)',
    'Meses_Contrato': 'Duração do Contrato (meses)',
    'Genero_Male': 'Gênero Masculino',
    'Idoso': 'Idoso (65+ anos)',
    'Tipo_Contrato_Mensal': 'Contrato Mensal',
    'Tipo_Contrato_One year': 'Contrato Anual',
    'Tipo_Contrato_Two year': 'Contrato Bienal (2 anos)',
    'Metodo_Pagamento_Electronic check': 'Pagamento por Cheque Eletrônico',
    'Metodo_Pagamento_Mailed check': 'Pagamento por Cheque Postal',
    'Metodo_Pagamento_Credit card (automatic)': 'Pagamento por Cartão de Crédito',
    'todo_Pagamento_Credit card (automatic)': 'Pagamento por Cartão de Crédito',
    'Tipo_Contrato_Bienal (2 anos)' : 'Contrato Bienal'
}


# Convertendo a coluna para numérico (caso ainda não esteja)
df_final['Cobranca_Total'] = pd.to_numeric(df_final['Cobranca_Total'], errors='coerce')

# Exibir as primeiras linhas para verificação
# df_final.head()


### GRÁFICOS E ANÁLISES

# VERIFICAÇÃO DE EVASÃO

# Calcular a porcentagem de evasão
total_clientes = df_final.shape[0]
total_evadidos = df_final[df_final['Evasao'] == 'Saíram da base'].shape[0]
porcentagem_evasao = (total_evadidos / total_clientes) * 100

# Definir o estilo do gráfico
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 6)

# Criar o gráfico
fig, ax = plt.subplots()

# Verificar a distribuição de Evasão de Clientes (churn)
grafico = sns.countplot(data=df_final, x='Evasao', palette='Set2', ax=ax)

# Títulos e rótulos
ax.set_title('Distribuição de Clientes Ativos vs Evadidos', fontsize=14)
ax.set_xlabel('Status do Cliente', fontsize=12)
ax.set_ylabel('Quantidade de Clientes', fontsize=12)

# Adicionar a caixa de texto explicativa dentro do gráfico
texto = f"Total de clientes: {total_clientes}\nEvasão: {porcentagem_evasao:.1f}%"
ax.text(0.7, 0.75, texto, transform=ax.transAxes, fontsize=12, ha='center', va='top',
        bbox=dict(facecolor='lightgray', edgecolor='gray', boxstyle='round,pad=1'))

# Adicionar números em cima de cada barra
for p in grafico.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 2, f'{int(height)}', ha='center', fontsize=12)

# Remover linhas de grade e bordas superior e direita
ax.grid(False)
sns.despine(top=True, right=True)

# Alterar o fundo do gráfico para um cinza claro
fig.patch.set_facecolor('lightgray')
ax.set_facecolor('whitesmoke')

# Exibir o gráfico
plt.tight_layout()
plt.show()


# ANÁLISE VARIÁVEIS CATEGÓRICAS

# Codificação das variáveis (como você já fez)
df_encoded = pd.get_dummies(df_final, columns=['Genero', 'Tipo_Contrato', 'Metodo_Pagamento'], drop_first=True)
df_encoded['Evasao'] = df_encoded['Evasao'].map({'Saíram da base': 1, 'Continuam na base': 0})


# # Calcular a correlação
correlacao = df_encoded.corr(numeric_only=True)[['Evasao']].sort_values(by='Evasao', ascending=False)

# # Renomear os índices usando o mapeamento
correlacao_renomeada = correlacao.rename(index=mapeamento_nomes)
correlacao_renomeada.head()
# Plotar o heatmap com os nomes em português
plt.figure(figsize=(12, 8))
sns.heatmap(correlacao_renomeada,
            annot=True,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            fmt=".2f",  # Formato com 2 casas decimais
            linewidths=.5)
plt.title('Correlação com Evasão de Clientes', pad=20)
plt.xlabel('Coeficiente de Correlação')
plt.ylabel('Variáveis Analisadas')
plt.yticks(rotation=0)  # Mantém os rótulos horizontais
plt.tight_layout()  # Ajusta o layout para evitar cortes
plt.show()


# EVASÃO POR TIPO DE CONTRATO
plt.figure(figsize=(12, 6))
sns.countplot(data=df_final, x='Tipo_Contrato', hue='Evasao', palette='Set2')
plt.title('Evasão por Tipo de Contrato')
plt.xlabel('Tipo de Contrato')
plt.ylabel('Quantidade de Clientes')
plt.legend(title='Evasão')
plt.grid(False)
sns.despine(top=True, right=True)
plt.show()

# MESES DE CONTRATOS QUE EVADIRAM 
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_final, x='Evasao', y='Meses_Contrato', palette='Set2')
plt.title('Distribuição de Meses em Contrato por Status de Evasão')
plt.xlabel('Status do Cliente')
plt.ylabel('Meses de Contrato')
plt.xticks([0, 1], ['Clientes Ativos', 'Clientes que Evadiram'])  # Melhorando as labels do eixo x
plt.show()


# EVASÃO POR MESES DE CONTRATO GRÁFICO 2
plt.figure(figsize=(12, 7))

# Criar o gráfico
ax = sns.histplot(data=df_final,
                 x='Meses_Contrato',
                 hue='Evasao',
                 kde=True,
                 palette='Set2',
                 bins=30,
                 alpha=0.7,
                 stat='count')  # Garante que mostre a contagem absoluta

# Personalizar título e labels
plt.title('Distribuição de Meses de Contrato por Status de Evasão',
          fontsize=16,
          pad=20)
plt.xlabel('Meses de Contrato', fontsize=12)
plt.ylabel('Total de Clientes', fontsize=12)  # Label corrigida



# Ajustes finais
plt.grid(axis='y', alpha=0.3)
sns.despine()
plt.tight_layout()
plt.show()



# EVASÃO DE CLIENTES QUE NÃO TIVERAM ACESSO AO SUPORTE
plt.figure(figsize=(10, 6))

# Filtrando apenas 'No' e 'Yes' e calculando a proporção
ct = pd.crosstab(df_final['Suporte_Tecnico'], df_final['Evasao'], normalize='index').loc[['No', 'Yes']]

# Renomeando os rótulos
ct = ct.rename(index={
    'Yes': 'Com acesso ao suporte',
    'No': 'Sem acesso ao suporte'
})

# Ordenando as categorias para melhor visualização
ct = ct.loc[['Com acesso ao suporte', 'Sem acesso ao suporte']]

# Plotando o gráfico
ax = ct.plot(
    kind='bar',
    stacked=True,
    color=['#66c2a5', '#fc8d62'],
    edgecolor='black',
    width=0.6
)

plt.title('Taxa de Evasão de clientes que tiveram \nAcesso ao Suporte Técnico', pad=20, fontsize=16)
plt.xlabel('')
plt.ylabel('Proporção')

# Formatando o eixo Y para porcentagem
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

# Removendo as bordas
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Ajustando a legenda
plt.legend(
    title='Evasão',
    labels=['Permaneceu', 'Cancelou'],
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.xticks(rotation=0)
plt.grid(False)
plt.tight_layout()

# Adicionando porcentagens nas barras
for n, x in enumerate([*ct.index.values]):
    for (proportion, y_loc) in zip(ct.loc[x], ct.loc[x].cumsum()):
        plt.text(x=n, y=(y_loc - proportion/2),
                s=f'{proportion:.1%}',
                color='white',
                fontsize=10,
                fontweight='bold',
                ha='center')

plt.show()


# EVASÃO POR MÉTODOS DE PAGAMENTOS

#Evasão por método de pagamentos
plt.figure(figsize=(12, 8))

# Criando o DataFrame com seus dados
data = {
    'Permaneceu': [1286, 1290, 1294, 1304],
    'Cancelou': [258, 232, 1071, 308]
}
df_payment = pd.DataFrame(data, index=[
    'Transferência Bancária',
    'Cartão de Crédito',
    'Cheque Eletrônico',
    'Cheque Postal'
])

# Convertendo para porcentagens
df_percent = df_payment.div(df_payment.sum(axis=1), axis=0) * 100

# Plotando barras horizontais empilhadas
ax = df_percent.plot(
    kind='barh',
    stacked=True,
    color=['#66c2a5', '#fc8d62'],
    edgecolor='black',
    width=0.7
)

plt.title('Evasão por Método de Pagamento (%)', fontsize=16, pad=20)
plt.xlabel('Proporção (%)', fontsize=12)
plt.ylabel('')

# Removendo bordas e linhas de grade
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(False)

# Ajustando a legenda
plt.legend(
    title='Status do Cliente',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    frameon=False
)

# Adicionando porcentagens nas barras
for i, (perm, canc) in enumerate(zip(df_percent['Permaneceu'], df_percent['Cancelou'])):
    # Para a parte "Permaneceu"
    if perm > 5:  # Só mostra se a porcentagem for significativa
        ax.text(perm/2, i, f"{perm:.1f}%", ha='center', va='center', color='white', fontweight='bold')

    # Para a parte "Cancelou"
    if canc > 5:  # Só mostra se a porcentagem for significativa
        ax.text(perm + canc/2, i, f"{canc:.1f}%", ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()


# FAIXA ETARIA DOS CANCELAMENTOS 
plt.figure(figsize=(8, 6))

# Criando o gráfico de barras
ax = sns.countplot(
    data=df_final, 
    x='Idoso', 
    hue='Evasao', 
    palette=['#66c2a5', '#fc8d62']
)

# Personalizando os rótulos do eixo X
plt.xticks(
    ticks=[0, 1], 
    labels=['Não Idosos', 'Idosos'],
    rotation=0
)

plt.title('Evasão por Faixa Etária', pad=20, fontsize=14)
plt.xlabel('Faixa Etária', fontsize=12)
plt.ylabel('Proporção de Clientes', fontsize=12)

# Ajustando a legenda
plt.legend(
    title='Status do Cliente',
    labels=['Permaneceu', 'Cancelou'],
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    frameon=False
)

# Removendo elementos visuais indesejados
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

# Calculando totais para cada categoria
totais = df_final['Idoso'].value_counts()

# Adicionando porcentagens nas barras
for p in ax.patches:
    # Calculando a porcentagem
    categoria = 0 if p.get_x() < 0.5 else 1  # 0=Não Idosos, 1=Idosos
    porcentagem = 100 * p.get_height() / totais[categoria]
    
    ax.annotate(
        f'{porcentagem:.1f}%', 
        (p.get_x() + p.get_width() / 2., p.get_height() / 2),  # Posiciona no meio da barra
        ha='center', 
        va='center',
        color='white',
        fontsize=10,
        fontweight='bold'
    )

plt.tight_layout()
plt.show()


# ANÁLISE VARIÁVEIS DE RISCO

dados_cancelamentos = {
    'Nível de Risco': [
        'Máximo Risco (4 fatores críticos)',
        'Alto Risco (3 fatores críticos)', 
        'Risco Moderado (2 fatores críticos)',
        'Risco Básico (1 fator crítico)'
    ],
    'Quantidade de Clientes': [120, 85, 60, 200]  # Valores fictícios
}

df_plot = pd.DataFrame(dados_cancelamentos)

# Descrição detalhada dos grupos de risco para o subtítulo
descricao_riscos = """
<b>Critérios de Risco:  Contrato Mensal, Sem Suporte, Cheque Eletrônico e Idoso<br>
"""

# Criar gráfico de barras horizontais
fig = px.bar(
    df_plot,
    y='Nível de Risco',
    x='Quantidade de Clientes',
    color='Nível de Risco',
    color_discrete_sequence=px.colors.sequential.Reds[1::2],
    title='<b>Distribuição de Cancelamentos por Nível de Risco</b><br><sup>'+descricao_riscos+'</sup>',
    labels={
        'Quantidade de Clientes': '<b>Número de Cancelamentos</b>',
        'Nível de Risco': '<b>Categoria de Risco</b>'
    },
    text='Quantidade de Clientes',
    orientation='h'
)

# Personalização avançada
fig.update_traces(
    marker_line_color='black',
    marker_line_width=1,
    textfont_size=11,
    textposition='outside',
    hovertemplate=(
        '<b>%{y}</b><br>' +
        'Total de cancelamentos: <b>%{x}</b><br>' +
        'Representa <b>%{customdata[0]:.1f}%</b> do total<extra></extra>'
    ),
    customdata=[(df_plot['Quantidade de Clientes']/df_plot['Quantidade de Clientes'].sum()*100).values.reshape(-1,1)]
)

# Layout final
fig.update_layout(
    uniformtext_minsize=10,
    uniformtext_mode='hide',
    xaxis_title='<b>Número de Clientes que Cancelaram</b>',
    yaxis_title='',
    yaxis={'categoryorder':'total ascending'},
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    margin={'l': 150, 'r': 20, 't': 160, 'b': 40},  # Ajuste para o subtítulo longo
    height=550  # Altura aumentada
)

fig.show()



# ==============================================
### Continuação chalen-telecom-x - desafio final
# ==============================================

# MATRIZ DE CORRELAÇÃO
df_corr = df_encoded.corr(numeric_only=True)

plt.figure(figsize=(16, 12))
sns.heatmap(
    df_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    vmin=-1, vmax=1,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.title("Matriz de Correlação entre Variáveis Numéricas", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# INSIGHT PRINCIPAL
correlacoes_evasao = df_corr['Evasao'].sort_values(ascending=False)
print("\nVariáveis mais correlacionadas com a Evasão:\n")
print(correlacoes_evasao)


# BOX PLOT: Tempo de Contrato vs Evasão
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_final,
    x='Evasao',
    y='Meses_Contrato',
    palette='Set2'
)

plt.title('Distribuição do Tempo de Contrato por Status de Evasão', fontsize=16)
plt.xlabel('Status do Cliente', fontsize=12)
plt.ylabel('Meses de Contrato', fontsize=12)
plt.xticks([0, 1], ['Continuam na base', 'Saíram da base'])

plt.tight_layout()
plt.show()


# BOX PLOT: Total Gasto vs Evasão
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_final,
    x='Evasao',
    y='Cobranca_Total',
    palette='Set2'
)

plt.title('Distribuição do Total Gasto por Status de Evasão', fontsize=16)
plt.xlabel('Status do Cliente', fontsize=12)
plt.ylabel('Cobrança Total (R$)', fontsize=12)
plt.xticks([0, 1], ['Continuam na base', 'Saíram da base'])

plt.tight_layout()
plt.show()



# SCATTER PLOT: Tempo de Contrato vs Total Gasto vs Evasão
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_final,
    x='Meses_Contrato',
    y='Cobranca_Total',
    hue='Evasao',
    palette='Set2',
    alpha=0.7
)

plt.title('Tempo de Contrato vs Total Gasto por Status de Evasão', fontsize=16)
plt.xlabel('Meses de Contrato', fontsize=12)
plt.ylabel('Cobrança Total (R$)', fontsize=12)
plt.legend(title='Evasão', loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# DIVISÃO DO CONJUNTO EM TREINO E TESTE
X = df_encoded.drop('Evasao', axis=1)
y = df_encoded['Evasao']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,  
    random_state=42, 
    stratify=y       
)

# Mostrar formatos das divisões
# print(f"Tamanho do conjunto de treino: {X_train.shape}")
# print(f"Tamanho do conjunto de teste: {X_test.shape}")
# print(f"Proporção de evasão no treino: {y_train.mean():.2%}")
# print(f"Proporção de evasão no teste: {y_test.mean():.2%}")



# CRIAÇÃO DOS MODELOS
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODELO 1: REGRESSÃO LOGÍSTICA
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)

y_pred_logistic = logistic_model.predict(X_test_scaled)


print("=== Regressão Logística ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred_logistic):.2%}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_logistic))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_logistic))


#  RANDOM FOREST (SEM NORMALIZAÇÃO)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Avaliar
print("\n=== Random Forest ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf):.2%}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_rf))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_rf))
