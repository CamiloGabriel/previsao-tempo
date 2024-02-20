import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Criar dados fictícios para simular uma previsão do tempo
dias = np.array([1, 2, 3, 4, 5, 6, 7])
temperaturas = np.array([22, 24, 23, 25, 26, 27, 28])

# Criar um DataFrame com os dados e fornecendo os nomes das features
df = pd.DataFrame({'Dia': dias, 'Temperatura': temperaturas})

# Vamos ajustar um modelo de regressão linear simples
modelo = LinearRegression()
modelo.fit(df[['Dia']], df['Temperatura'])

# Prever a temperatura para o dia seguinte
dia_seguinte = np.array([[8]])
previsao = modelo.predict(dia_seguinte)

print(f"Previsão da temperatura para o dia seguinte:, {previsao[0]:.2f}")
    
# Criar um gráfico para visualizar a tendência
plt.scatter(dias, temperaturas, color='b')
plt.plot(dias, temperaturas, color='r')
plt.xlabel('Dia')
plt.ylabel('Temperatura')
plt.show()
