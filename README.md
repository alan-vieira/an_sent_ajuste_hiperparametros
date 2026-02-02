# 📈 Otimização de Hiperparâmetros com GridSearchCV

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Data Science](https://img.shields.io/badge/Data%20Science-Optimization-blue?style=for-the-badge)

## 📖 Descrição
Este repositório é dedicado ao **Fine-tuning** de modelos de Machine Learning para classificação de sentimentos. Através da técnica de **GridSearchCV**, explorei o espaço de parâmetros de diversos algoritmos para encontrar a configuração ótima, garantindo a máxima performance preditiva.

## 🚀 Como funciona o GridSearchCV
O `GridSearchCV` realiza uma busca exaustiva sobre uma grade de parâmetros especificada, combinada com **Validação Cruzada (Cross-Validation)**. 

### Exemplo Prático: Support Vector Classifier (SVC)
Para o modelo SVC, definimos diferentes valores para os parâmetros `C`, `gamma` e `kernel`:

```python
# Definição do Grid de Parâmetros
parameters = {
    'C': [1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.001, 0.0001],
    'kernel': ['linear', 'rbf']
}

# Pipeline integrando Vetorização e Modelo
modelo = Pipeline(steps=[
    ('vectorizer', TfidfVectorizer()),
    ('modelo', SVC())
])

# Instanciação do GridSearchCV
clf = GridSearchCV(modelo, parameters, refit='accuracy', verbose=3)
```

## 🧮 A Matemática do Treinamento

O número total de treinamentos é o produto das combinações de parâmetros pelo número de folds da validação cruzada:

**4 (C) x 4 (gamma) x 2 (kernel) x 5 (folds) = 160 fits**

## 📊 Resultados e Rankeamento

O processo gera um ranking detalhado. No caso do SVC, o modelo vencedor utilizou:

**Acurácia**: 0.7895

**Parâmetros**: `{'C': 10, 'gamma': 1, 'kernel': 'rbf'}`

| Rank | Parâmetros (C, gamma, kernel)                     | Score (Mean Test) |
|:-----|:--------------------------------------------------|------------------:|
| 1º   | {'C': 10, 'gamma': 1, 'kernel': 'rbf'}            |            0.7895 |
| 2º   | {'C': 1000, 'gamma': 1, 'kernel': 'rbf'}          |            0.7890 |
| 3º   | {'C': 100, 'gamma': 1, 'kernel': 'rbf'}           |            0.7890 |

## 📂 Modelos Otimizados neste Projeto

Acesse os notebooks específicos para cada implementação:

[🔥 Gradient Boosting](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/gradient_boosting_tfidf_oversampling.ipynb)

[🌲 Random Forest](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/random_forest_tfidf_oversampling.ipynb)

[🤖 XGBoost](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/xgb_tfidf_oversampling.ipynb)

[📈 Logistic Regression](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/logistic_regression_tfidf_oversampling.ipynb)

[🧠 MLP (Rede Neural)](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/mlp_tfidf_oversampling.ipynb)

[👉 Veja todos os 10 modelos no repositório](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros)

## 🛠️ Ferramentas

`Python` • `Scikit-Learn` • `Pandas` • `Spacy` • `Simplemma` • `Matplotlib` • `Seaborn`

## 👤 Autor

**Alan Vieira** - *Engenheiro de Telecomunicações & Especialista em Dados*

- [LinkedIn](https://www.linkedin.com/in/alansilvavieira)

- [GitHub Portfólio](https://github.com/alan-vieira)





