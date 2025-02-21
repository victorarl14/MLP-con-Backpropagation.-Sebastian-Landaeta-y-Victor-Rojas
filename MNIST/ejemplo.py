import numpy as np
import gzip
from PIL import Image

# Función para cargar imágenes EMNIST
def cargar_imagenes(ruta):
    with gzip.open(ruta, 'rb') as f:
        # Leer el encabezado del archivo
        magic = int.from_bytes(f.read(4), 'big')
        num_imagenes = int.from_bytes(f.read(4), 'big')
        filas = int.from_bytes(f.read(4), 'big')
        columnas = int.from_bytes(f.read(4), 'big')
        
        # Leer los datos de las imágenes
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_imagenes, filas, columnas)
        
        # Rotar y voltear las imágenes (EMNIST está en formato invertido)
        data = np.transpose(data, (0, 2, 1))
        data = np.flip(data, axis=2)
        
        return data

# Función para cargar etiquetas EMNIST
def cargar_etiquetas(ruta):
    with gzip.open(ruta, 'rb') as f:
        # Leer el encabezado del archivo
        magic = int.from_bytes(f.read(4), 'big')
        num_etiquetas = int.from_bytes(f.read(4), 'big')
        
        # Leer las etiquetas
        etiquetas = np.frombuffer(f.read(), dtype=np.uint8)
        
        return etiquetas

# Cargar los datos de EMNIST Balanced
print("Cargando datos de EMNIST Balanced...")
X_train = cargar_imagenes('emnist-balanced-train-images-idx3-ubyte.gz')
y_train = cargar_etiquetas('emnist-balanced-train-labels-idx1-ubyte.gz')
X_test = cargar_imagenes('emnist-balanced-test-images-idx3-ubyte.gz')
y_test = cargar_etiquetas('emnist-balanced-test-labels-idx1-ubyte.gz')

# Normalizar las imágenes
X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

# Clase MLP (Perceptrón Multicapa)
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        # Inicializar pesos y sesgos
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2./input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2./hidden_size)
        self.b2 = np.zeros(output_size)
        self.lr = lr

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        # Propagación hacia adelante
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.softmax(self.z2)

    def compute_loss(self, y_pred, y_true):
        # Calcular la pérdida (entropía cruzada)
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[np.arange(m), y_true])
        return np.sum(log_likelihood) / m

    def backward(self, X, y_true, y_pred):
        # Retropropagación
        m = y_true.shape[0]
        
        # Gradiente de la capa de salida
        grad_z2 = y_pred.copy()
        grad_z2[np.arange(m), y_true] -= 1
        grad_z2 /= m
        
        # Gradientes de la capa oculta
        grad_W2 = np.dot(self.a1.T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0)
        grad_a1 = np.dot(grad_z2, self.W2.T)
        grad_z1 = grad_a1 * (self.z1 > 0)
        
        # Gradientes de la capa de entrada
        grad_W1 = np.dot(X.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0)
        
        # Actualizar pesos y sesgos
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

    def train(self, X, y, epochs, batch_size=32):
        # Entrenamiento del modelo
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            
            total_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_pred, y_batch)
                self.backward(X_batch, y_batch, y_pred)
                
                total_loss += loss * len(X_batch)
            
            print(f'Época {epoch+1}/{epochs}, Pérdida: {total_loss/len(X):.4f}')

    def predict(self, X):
        # Predicción
        return np.argmax(self.forward(X), axis=1)

# Crear y entrenar el modelo
print("Creando y entrenando el modelo...")
modelo = MLP(784, 128, 47, lr=0.1)  # 47 clases en EMNIST Balanced
modelo.train(X_train, y_train, epochs=20, batch_size=64)

# Evaluar en el conjunto de prueba
precision = np.mean(modelo.predict(X_test) == y_test)
print(f'Precisión en test: {precision:.4f}')

# Función para preprocesar una imagen de entrada
def preprocesar_imagen(ruta):
    img = Image.open(ruta).convert('L')  # Convertir a escala de grises
    img = img.resize((28, 28))  # Redimensionar a 28x28
    img_array = 255 - np.array(img)  # Invertir colores (fondo blanco -> fondo negro)
    img_array = img_array.astype(np.float32) / 255.0  # Normalizar
    return img_array.reshape(1, -1)  # Aplanar a un vector de 784 elementos

# Ejemplo de predicción
ruta_imagen = 'F.png'  # Reemplaza con la ruta de tu imagen
imagen_preprocesada = preprocesar_imagen(ruta_imagen)
prediccion = modelo.predict(imagen_preprocesada)

# Mapear la predicción a un carácter
caracteres = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
print(f'El carácter es: {caracteres[prediccion[0]]}')