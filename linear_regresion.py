from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def linear_regresion(test_size=0.2, random_state=42):
    """
    Entrena un modelo de regresión lineal usando la característica BMI del dataset de Diabetes.

    Parámetros:
    - test_size: proporción del conjunto de prueba (default 0.2)
    - random_state: semilla para la reproducibilidad (default 42)

    Retorna:
    - mse: Error Cuadrático Medio del conjunto de prueba
    - r2: coeficiente de determinación R² del conjunto de prueba
    """

    # 1. Cargar el dataset
    diabetes = load_diabetes()
    
    # 2. Seleccionar la característica BMI (columna índice 2)
    X = diabetes.data[:, 2].reshape(-1, 1)  # Necesitamos matriz 2D
    y = diabetes.target

    # 3. Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. Entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Realizar predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # 6. Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 7. Devolver resultados como tupla
    return mse, r2
