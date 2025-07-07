# 📊 Relatório de Análise de Evasão de Clientes – Telecom X

## Introdução

Este relatório tem como objetivo apresentar a análise exploratória dos dados da empresa **Telecom X**, com foco na **evasão de clientes (churn)**. Através da exploração visual dos dados, foram identificados padrões que ajudam a compreender os principais motivos que levam os clientes a deixarem a empresa.

Os gráficos utilizados neste relatório foram gerados a partir do script disponível no repositório:

> [https://github.com/gustavor8/challenge-alura-telecom-x](https://github.com/gustavor8/challenge-alura-telecom-x)

---

## Gráficos Utilizados

### 1. Distribuição do Tipo de Contrato entre Clientes Ativos e Evasivos

- **Gráfico**: Gráfico de barras
- **Observação**: A maior parte dos clientes que cancelaram seus serviços possuíam **contratos mensais**.

### 2. Distribuição de Evasão por Tempo de Contrato (Tenure)

- **Gráfico**: Histograma
- **Observação**: A evasão é mais comum nos **primeiros meses de contrato**, especialmente até o 12º mês.

### 3. Acesso ao Suporte Técnico e Evasão

- **Gráfico**: Gráfico de barras (clusterizado por churn)
- **Observação**: Clientes que **não utilizaram o suporte técnico** apresentaram maior taxa de evasão.

### 4. Método de Pagamento e Evasão

- **Gráfico**: Gráfico de barras
- **Observação**: A **maior taxa de churn foi identificada entre clientes que pagam por cheque eletrônico**.

### 5. Idade e Evasão

- **Gráfico**: Gráfico de pizza ou barras, categorizando por faixa etária e churn
- **Observação**: Há **uma taxa de evasão mais elevada entre clientes idosos**.

---

## Conclusão

Com base nos gráficos e nas análises visuais, é possível traçar um perfil do cliente com **maior propensão à evasão**:

- **Tipo de Contrato**: Mensal – menor fidelização e maior facilidade de cancelamento.
- **Tempo de Contrato**: Clientes recentes (com poucos meses de permanência) ainda não criaram vínculo com a empresa.
- **Suporte Técnico**: A ausência de interação com o suporte pode indicar **falta de engajamento**, ou que os clientes enfrentam problemas sem buscar resolução, o que pode motivar o cancelamento.
- **Método de Pagamento**: Cheque eletrônico – possivelmente associado a clientes com menor acesso digital ou instabilidade financeira.
- **Idade**: Clientes idosos têm uma taxa de cancelamento superior, podendo indicar dificuldade de uso dos serviços ou insatisfação com atendimento/acesso digital.

---

## Recomendações

Com base nas conclusões, sugere-se:

1. **Incentivar contratos de maior duração**, oferecendo descontos ou benefícios progressivos.
2. **Investir em onboarding e engajamento nos primeiros meses** de contrato.
3. **Promover o suporte técnico de forma ativa**, mostrando seus benefícios logo no início do relacionamento.
4. **Rever políticas relacionadas ao pagamento por cheque eletrônico**, oferecendo alternativas mais vantajosas e acessíveis.
5. **Criar canais exclusivos de atendimento e suporte digital simplificado para idosos**.

---

---

# SEGUNDA PARTE DO DESAFIO

---

---

# 🧩 Modelagem Preditiva e Análise de Importância das Variáveis -

## Modelos Utilizados

Foram treinados dois modelos complementares para prever a evasão de clientes:

- **Regressão Logística:** Um modelo linear, escolhido por ser interpretável. Exige normalização das variáveis numéricas para equilibrar suas escalas, garantindo que nenhuma feature tenha peso desproporcional.
- **Random Forest:** Um modelo de árvore de decisão em conjunto (ensemble), robusto para dados categóricos e numéricos. Não exige normalização, pois as árvores avaliam condições de corte por limiares.

A divisão dos dados foi de **70% para treino** e **30% para teste**, garantindo a **estratificação da variável alvo**, mantendo a proporção original de evasão.

---

## Avaliação de Desempenho

**Métricas principais utilizadas:**

- **Acurácia:** Proporção geral de acertos.
- **Precisão:** Proporção de acertos nos casos previstos como evasão.
- **Recall:** Capacidade de capturar todos os clientes que realmente evadiram.
- **F1-Score:** Equilíbrio entre Precisão e Recall.
- **Matriz de Confusão:** Distribuição de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.

Os resultados indicaram que:

- **Random Forest apresentou maior Recall** em relação à Reg Logística, detectando mais casos de evasão, com ligeiro risco de falsos positivos.
- **Regressão Logística teve precisão alta**, mas recall mais baixo, indicando maior segurança ao prever churn, porém menor abrangência.

---

## Análise das Variáveis

**Regressão Logística:**  
A análise dos **coeficientes** mostrou que:

- Contrato Mensal, ausência de suporte técnico, método de pagamento por cheque eletrônico e ser idoso **aumentam a probabilidade de evasão**.
- Contratos de maior duração ou pagamentos automáticos reduzem o risco.

**Random Forest:**  
A árvore indicou como **principais variáveis**:

- **Tempo de Contrato:** fator de maior peso na divisão das árvores.
- **Tipo de Contrato:** contratos mensais são mais propensos a churn.
- **Método de Pagamento:** cheque eletrônico destacou-se como forte preditor.
- **Suporte Técnico:** ausência impacta diretamente.

Essas variáveis concentram as maiores contribuições para a redução da impureza (Gini).

---

## Conclusão da Modelagem

Os modelos confirmam os padrões observados na exploração inicial:

- **Combinar contrato mensal, baixo tempo de permanência, cheque eletrônico e ausência de suporte técnico** configura o perfil de **alto risco**.
- É recomendável usar essas variáveis como **pilares para campanhas de retenção**, direcionando ofertas, comunicações e suporte personalizado.

**Próximos passos sugeridos:**

- Avaliar outros algoritmos como KNN e SVM para ampliar comparações.
- Ajustar hiperparâmetros para reduzir possíveis sinais de overfitting.
- Implantar o modelo em produção com monitoramento periódico do desempenho.

---
