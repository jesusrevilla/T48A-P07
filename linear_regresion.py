# Paso 1: Importar las librerías necesarias
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def linear_regresion():
    # Paso 2: Cargar el dataset de diabetes
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Paso 3: Usar solo una variable (columna 2: índice de masa corporal)
    X = X[:, np.newaxis, 2]

    # Paso 4: Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Paso 5: Ajustar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Paso 6: Predicciones
    y_pred = model.predict(X_test)

    # Paso 7: Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Redondear a dos cifras y devolver
    return round(mse, 2), round(r2, 2)

# ---- Uso de la función ----
mse, r2 = linear_regresion()
print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R²):", r2)
