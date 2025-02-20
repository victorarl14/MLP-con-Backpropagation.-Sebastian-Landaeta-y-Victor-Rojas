import numpy as np
import gzip
import os
from PIL import Image

# Cargar imágenes y etiquetas
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows * cols)

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Clase MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
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
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.softmax(self.z2)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[np.arange(m), y_true])
        return np.sum(log_likelihood) / m

    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]
        
        # Gradiente de la capa de salida
        grad_z2 = y_pred.copy()
        grad_z2[np.arange(m), y_true] -= 1
        grad_z2 /= m
        
        # Gradientes capa oculta
        grad_W2 = np.dot(self.a1.T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0)
        
        grad_a1 = np.dot(grad_z2, self.W2.T)
        grad_z1 = grad_a1 * (self.z1 > 0)
        
        # Gradientes capa entrada
        grad_W1 = np.dot(X.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0)

        # Actualizar parámetros
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

    def train(self, X, y, epochs, batch_size=32):
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
        return np.argmax(self.forward(X), axis=1)

# Preprocesar imagen de entrada
def preprocesar_imagen(ruta):
    img = Image.open(ruta).convert('L').resize((28, 28))
    img_array = 255 - np.array(img)  # Invertir colores
    return img_array.astype(np.float32).reshape(1, -1) / 255.0

X_train = load_images('train-images-idx3-ubyte.gz')
y_train = load_labels('train-labels-idx1-ubyte.gz')
X_test = load_images('t10k-images-idx3-ubyte.gz')
y_test = load_labels('t10k-labels-idx1-ubyte.gz')

# Normalizar datos
X_train = X_train / 255.0
X_test = X_test / 255.0

# Crear y entrenar modelo
modelo = MLP(784, 128, 10, lr=0.1)
modelo.train(X_train, y_train, epochs=10, batch_size=64)

# Evaluar en test
precision = np.mean(modelo.predict(X_test) == y_test)
print(f'Precisión en test: {precision:.4f}')

# Ejemplo de predicción (reemplazar 'digito.png' con tu imagen)
imagen = preprocesar_imagen('1.png')
prediccion = modelo.predict(imagen)
print(f'El dígito es: {prediccion[0]}')