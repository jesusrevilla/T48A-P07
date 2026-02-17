import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def linear_regresion():
 # Cargar el dataset
    diabetes = load_diabetes()
    X = diabetes.data[:, 2].reshape(-1, 1)  # Solo BMI (índice 2)
    y = diabetes.target
 
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predecir y calcular el error y R2
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2
