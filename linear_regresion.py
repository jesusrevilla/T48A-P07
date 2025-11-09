from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
def linear_regresion():
    #Cargar datos diabetes
    diabetes = load_diabetes()
    #Seleccionar solo la segunda característica (BMI - índice 2) de lo contrario tendremos otros datos
    X = diabetes.data[:, 2:3]
    y = diabetes.target
    #Division de datos para 80% entrenamiento, 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    #Realizar predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)
    #Calcular el error cuadratico medio y el coeficiente de determinación R^2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) 
    #Redondear a 2 decimales como lo pide la prueba
    mse_redondeado = round(mse, 2)
    r2_redondeado = round(r2, 2)
    print(f"MSE: {mse_redondeado}")
    print(f"R2: {r2_redondeado}")
    #Regresar como tupla
    return (mse_redondeado, r2_redondeado)


resultado = linear_regresion()
print(f"Resultado: {resultado}")
