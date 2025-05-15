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
