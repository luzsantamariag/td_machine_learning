# Importar librerias
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


# Cargar dataset
iris = load_iris()

# Mostrar los primeros registros del dataset
print(iris.data[:5])

# Mostrar los nombres de las caracteristicas
print(iris.feature_names)

# Mostrar los nombres de las clases
print(iris.target_names)

# Mostrar los valores de la variable objetivo
print(iris.target[:5])

# Mostrar la descripción del dataset
print(iris.DESCR)

# Dividir dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Crear modelo de clasificacion
clf = DecisionTreeClassifier(random_state=42)

# Entrenar modelo
clf.fit(X_train, y_train)

# Predecir
y_pred = clf.predict(X_test)

# Evaluar
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
