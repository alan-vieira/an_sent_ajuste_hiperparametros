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

4 (C) x 4 (gamma) x 2 (kernel) x 5 (folds) = 160 fits

## Acesso ao projeto

Você pode acessar os códigos fonte dos projetos ou [baixá-los](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/archive/refs/heads/main.zip).

[Gradient Boosting](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/gradient_boosting_tfidf_oversampling.ipynb)

[kNN (K-Nearest Neighbors)](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/knn_tfidf_oversampling.ipynb)

[Logistic Regression](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/logistic_regression_tfidf_oversampling.ipynb)

[MLP (Multi Layer Perceptron)](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/mlp_tfidf_oversampling.ipynb)

[MultinomialNB](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/multinomialnb_tfidf_oversampling.ipynb)

[Passive Aggressive](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/passive_aggressive_tfidf_oversampling.ipynb)

[Random Forest](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/random_forest_tfidf_oversampling.ipynb)

[SGD (Stochastic Gradient Descent)](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/sgd_tfidf_oversampling.ipynb)

[SVC (Support Vector Classification)](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/svc_tfidf_oversampling.ipynb)

[XGB (XGBoost)](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/blob/main/xgb_tfidf_oversampling.ipynb)

## Abrir e rodar o projeto
Após baixado, para o funcionamento correto da aplicação as seguintes dependêcias deverão ser instaladas.

- `pandas`
- `nltk`
- `pip setuptools wheel`
- `spacy`
- `pt_core_news_sm`
- `simplemma`
- `searchgrid`

## Autor

| [<img src="https://avatars.githubusercontent.com/alan-vieira" width=115><br><sub>Alan Vieira</sub>](https://github.com/alan-vieira) |
| :---: |
