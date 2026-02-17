import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def linear_regresion():
    # 1. Cargar el conjunto de datos de la Diabetes
    diabetes = load_diabetes()
    
    # 2. Seleccionar una sola característica (BMI, índice 2) para X
    # El índice 2 corresponde a la característica 'bmi'.
    # Se utiliza np.newaxis para reformatear la característica a una matriz 2D (n_muestras, 1),
    # que es el formato que sklearn espera para X.
    X = diabetes.data[:, np.newaxis, 2]
    y = diabetes.target
    
    # 3. Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
    # Usaremos el 75% para entrenamiento y el 25% para prueba (valor predeterminado de test_size).
    # Un random_state se utiliza para asegurar la reproducibilidad de la división.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Entrenar un modelo de LinearRegression con los datos de entrenamiento
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. Realizar predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # 6. Calcular el Error Cuadrático Medio (MSE) y el coeficiente de determinación (R²)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 7. Regresarlos como una tupla: (mse, r2)
    return (mse, r2)

if __name__ == '__main__':
    # Ejecución de prueba para verificar los resultados
    mse, r2 = linear_regresion()
    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"Coeficiente de Determinación (R²): {r2:.4f}")
    
    # Los valores obtenidos con random_state=42 y test_size=0.25 son:
    # MSE: 4060.27
    # R²: 0.2510

