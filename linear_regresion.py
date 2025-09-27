from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def linear_regresion(test_size=0.2, random_state=42):
    """
    Carga el dataset de diabetes, usa la característica BMI (índice 2),
    entrena una regresión lineal y devuelve (mse, r2).
    """
    # Cargar dataset y seleccionar BMI (columna índice 2)
    diabetes = load_diabetes()
    X = diabetes.data[:, 2].reshape(-1, 1)  # BMI como matriz 2D (n_samples, 1)
    y = diabetes.target

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predecir sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return (mse, r2)
