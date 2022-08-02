# Ajuste de Hiperparametros com GridSearchCV
## Descrição do projeto
Ajuste de hiperparâmetros de modelos utilizados para classificação de sentimentos, com auxílio do GridSearchCV. 

## Funcionalidades do projeto

- `Funcionalidade 1`: Treina os modelos com combinação de vários paramentos, para encontrar os melhores.
- `Funcionalidade 2`: A partir de algumas avaliações, gera um hanking com a melhor configuração de hiperparâmetros.

## Aplicação
O GridSearchCV é uma técnica que combina exaustivamente todos os hiperparâmentos que passamos, referentes ao modelo, para que, por meio de uma avaliação cruzada encontremos o melhor ajuste.

Vamos tomar com exemplo o classificador Support Vector Classifier, onde passamos os parametros C, gamma e kernel.

```python
parameters = {'C':[1, 10, 100, 1000],
              'gamma':[1, 0.1, 0.001, 0.0001],
              'kernel':['linear', 'rbf']}
```

E por meio de um pipeline informamdos o vetorizador, que nesse caso foi o Tfidf, seguido do modelo a ser avaliado.

```python
modelo = Pipeline(steps=[
             ('vectorizer' , TfidfVectorizer()),
            ('modelo', SVC())
            ])
```
 
 Agora instanciamos o GridSearchCV como clf, declarando o modelo, os parametros, no refit informamos a métrica de avaliação que aparecerá no rankeamento e o verbose é uma opição aparecer o andamento do treinamento.
 
 ```python
clf = GridSearchCV(modelo, parameters, refit = 'accuracy', verbose=3)
```

Declarando os dados de treino para o GridSearchCV avaliar par rankear o melhor conjundo de parâmetros.

 ```python
clf.fit(X_train, y_train).best_score_
```

Fitting 5 folds for each of 32 candidates, totalling 160 fits
[CV 1/5] END modelo__C=1, modelo__gamma=1, modelo__kernel=linear;, score=0.718 total time=  24.1s
[CV 2/5] END modelo__C=1, modelo__gamma=1, modelo__kernel=linear;, score=0.714 total time=  18.8s
[CV 3/5] END modelo__C=1, modelo__gamma=1, modelo__kernel=linear;, score=0.712 total time=  15.0s
[CV 4/5] END modelo__C=1, modelo__gamma=1, modelo__kernel=linear;, score=0.731 total time=  15.2s
[CV 5/5] END modelo__C=1, modelo__gamma=1, modelo__kernel=linear;, score=0.729 total time=  15.1s
[CV 1/5] END modelo__C=1, modelo__gamma=1, modelo__kernel=rbf;, score=0.787 total time=  19.8s
[CV 2/5] END modelo__C=1, modelo__gamma=1, modelo__kernel=rbf;, score=0.771 total time=  21.4s
[CV 3/5] END modelo__C=1, modelo__gamma=1, modelo__kernel=rbf;, score=0.768 total time=  19.6s
[CV 4/5] END modelo__C=1, modelo__gamma=1, modelo__kernel=rbf;, score=0.793 total time=  19.9s
[CV 5/5] END modelo__C=1, modelo__gamma=1, modelo__kernel=rbf;, score=0.788 total time=  19.9s
[CV 1/5] END modelo__C=1, modelo__gamma=0.1, modelo__kernel=linear;, score=0.718 total time=  15.0s
[CV 2/5] END modelo__C=1, modelo__gamma=0.1, modelo__kernel=linear;, score=0.714 total time=  15.1s
[CV 3/5] END modelo__C=1, modelo__gamma=0.1, modelo__kernel=linear;, score=0.712 total time=  15.0s

## Ferramentas utilizadas
- `Jupyter Notebook`
- `Python`
- `Sklearn`

## Acesso ao projeto

Você pode acessar o [código fonte do projeto]() ou [baixá-lo](https://github.com/alan-vieira/an_sent_ajuste_hiperparametros/archive/refs/heads/main.zip).

## Abrir e rodar o projeto
Após baixado, para o funcionamento correto da aplicação as seguintes dependêcias deverão ser instaladas.

- `pandas`
- `nltk`

## Autor

| [<img src="https://avatars.githubusercontent.com/alan-vieira" width=115><br><sub>Alan Vieira</sub>](https://github.com/alan-vieira) |
| :---: |
