# Predictive Health — Predição de Diabetes

Projeto de Machine Learning voltado à análise de dados de saúde e à comparação de algoritmos de classificação para a predição de diabetes.

## 🎯 Objetivo

Explorar uma base de dados sobre diabetes, aplicar técnicas de pré-processamento e comparar diferentes modelos de Machine Learning utilizando métricas de classificação e validação cruzada.

## 📊 Base de Dados

A base utilizada contém 768 registros, 8 variáveis de entrada e uma variável-alvo chamada `Outcome`.

As variáveis de entrada são:

- `Pregnancies`: número de gestações;
- `Glucose`: nível de glicose;
- `BloodPressure`: pressão arterial;
- `SkinThickness`: espessura da dobra cutânea;
- `Insulin`: nível de insulina;
- `BMI`: índice de massa corporal (IMC);
- `DiabetesPedigreeFunction`: indicador relacionado ao histórico familiar de diabetes;
- `Age`: idade.

A variável `Outcome` representa a classe registrada na base: 0 ou 1.

## ⚙️ Etapas do Projeto

1. Carregamento e exploração dos dados.
2. Identificação e substituição por valores ausentes dos zeros considerados inválidos em determinadas variáveis.
3. Divisão dos dados em conjuntos de treinamento e teste, com estratificação.
4. Tratamento dos valores ausentes por imputação da mediana dentro dos pipelines de cada modelo.
5. Padronização das variáveis para os algoritmos que se beneficiam desse procedimento.
6. Treinamento e comparação de oito modelos de classificação.
7. Avaliação por meio de acurácia, precisão, sensibilidade (recall), F1-score, ROC-AUC e validação cruzada estratificada.
8. Exportação dos resultados da comparação para um arquivo CSV.

## 🤖 Modelos de Machine Learning

- Dummy Classifier (modelo de referência);
- Regressão Logística;
- K-Nearest Neighbors (KNN);
- Support Vector Machine (SVM);
- Rede Neural (MLP);
- Árvore de Decisão;
- Random Forest (Floresta Aleatória);
- Gradient Boosting.

## 📈 Resultados Preliminares

Resultados obtidos na divisão atual dos dados, com 80% para treinamento e 20% para teste, utilizando `random_state=42`.

| Modelo | Acurácia no teste | F1-score | ROC-AUC |
|---|---:|---:|---:|
| Árvore de Decisão | 78,57% | 0,6916 | 0,7887 |
| Random Forest | 76,62% | 0,6400 | 0,8210 |
| Gradient Boosting | 75,32% | 0,6122 | 0,8309 |
| KNN | 75,32% | 0,6346 | 0,7899 |
| SVM | 74,03% | 0,6000 | 0,7964 |
| Rede Neural (MLP) | 72,73% | 0,6250 | 0,8031 |
| Regressão Logística | 70,78% | 0,5455 | 0,8130 |
| Dummy Classifier | 59,74% | 0,3922 | 0,5452 |

A Árvore de Decisão apresentou a maior acurácia e o maior F1-score nesse conjunto de teste. Já o Gradient Boosting obteve o maior ROC-AUC.

Esses resultados são preliminares e refletem a configuração atual dos experimentos. O desempenho pode variar conforme a divisão dos dados e os parâmetros dos modelos.

## 🗂️ Estrutura do Projeto

```text
Predictive_Health/
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── 01_exploracao.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── explore_data.py
│   └── evaluation.py
├── outputs/
│   └── resultados_modelos.csv
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ▶️ Como Executar

1. Clone ou baixe o repositório.
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Execute o projeto a partir da pasta raiz:

   ```bash
   python main.py
   ```

Os resultados serão exibidos no terminal e salvos em `outputs/resultados_modelos.csv`.

## ⚠️ Limitações

Este é um projeto educacional de Machine Learning, não uma ferramenta de diagnóstico médico. Os resultados não comprovam validade clínica nem relações de causa e efeito.

Como o conjunto de teste já foi utilizado para comparar os modelos, uma avaliação final independente será necessária para obter uma estimativa mais confiável da capacidade de generalização.

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook