import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def linear_regresion():
    pass
    # Cargar el dataset
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Seleccionamos solo la característica BMI (índice 2)
    X = X[:, np.newaxis, 2]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Devolver resultados 
    return round(mse, 2), round(r2, 2)

# Uso de la función
mse, r2 = linear_regresion()
print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R²):", r2)
