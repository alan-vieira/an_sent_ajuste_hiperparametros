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
