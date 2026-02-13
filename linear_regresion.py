import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def linear_regresion():
    #Carga el conjunto de datos de la Diabetes desde sklearn.datasets
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    #Seleciona una sola característica, la segunda característica, variable independiente X
    X_single_feature = X[:, np.newaxis, 2] #Se usa para seleccionar el índice 2 y reformar el array
    
    #Divide el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_single_feature, y, test_size=0.2, random_state=42 #SE usa 42 para mayor cambio y efectividad
    )
    
    #Entrena el modelo con los datos de entrenamiento
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    #Predice sobre el conjunto de prueba
    y_pred = model.predict(X_test)
    
    #Calcula el (MSE) y (R2) para evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return (mse, r2)

#Se ejecuta la función para obtener los resultados y verificarlos
mse_result, r2_result = linear_regresion()
print(f"Error Cuadrático Medio (MSE): {mse_result:.2f}")
print(f"Coeficiente de Determinación (R²): {r2_result:.2f}")
