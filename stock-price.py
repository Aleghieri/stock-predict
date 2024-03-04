# 1. Importando as bibliotecas necessárias.
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# 2. Definindo a função para preparar os dados para o modelo de regressão linear.
def prepare_data(df, forecast_col, forecast_out, test_size):
    # Criando a coluna de labels com os valores deslocados para frente.
    label = df[forecast_col].shift(-forecast_out)
    # Selecionando apenas a coluna de interesse e convertendo para um array numpy.
    X = np.array(df[[forecast_col]])
    # Normalizando os dados.
    X = preprocessing.scale(X)
    # Selecionando os dados que serão usados para previsão futura.
    X_lately = X[-forecast_out:]
    # Removendo os dados que serão usados para previsão futura do conjunto X.
    X = X[:-forecast_out]
    # Removendo os NaNs da coluna de labels.
    label.dropna(inplace=True)
    # Convertendo a coluna de labels para um array numpy.
    y = np.array(label)
    # Dividindo os dados em conjuntos de treinamento e teste.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    # Retornando os dados preparados.
    response = [X_train, X_test, y_train, y_test, X_lately]
    return response

# 3. Carregando os dados do arquivo CSV.
df = pd.read_csv(r"C:\Users\danie\Downloads\MGLUY.csv")

# 4. Definindo as variáveis para previsão.
forecast_col = 'Close'
forecast_out = 5
test_size = 0.2

# 5. Preparando os dados e aplicando Machine Learning.
X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size)
learner = LinearRegression()
learner.fit(X_train, Y_train)

# 6. Realizando a previsão dos preços.
score = learner.score(X_test, Y_test)
forecast = learner.predict(X_lately)

response = {}
response['test_score'] = score
response['forecast_set'] = forecast

# 7. Resultado
print(response)
