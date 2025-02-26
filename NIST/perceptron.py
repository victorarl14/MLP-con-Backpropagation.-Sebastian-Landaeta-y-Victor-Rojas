import time
import numpy as np
import cv2
import os

# Configuración de la red neuronal
INPUT_SIZE = 28 * 28  # Imágenes de 28x28
HIDDEN_SIZE = 256
OUTPUT_SIZE = 62  # Establecer OUTPUT_SIZE a 62 (fijo)
LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 64

# Cargar datos desde la carpeta by_class
def load_nist19(dataset_path):
    images = []
    labels = []
    class_mapping = {}  # Diccionario para asignar índices a cada carpeta (clase)

    # Validar que la carpeta exista
    if not os.path.exists(dataset_path):
        raise ValueError(f"❌ ERROR: La carpeta '{dataset_path}' no existe.")

    # Obtener lista de subcarpetas (cada una representa una clase)
    class_folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])

    if not class_folders:
        raise ValueError(f"❌ ERROR: No se encontraron subcarpetas dentro de '{dataset_path}'.")

    print(f"📂 Detectadas {OUTPUT_SIZE} clases (fijas).")

    # Asignar un índice a cada clase
    for idx, class_folder in enumerate(class_folders):
        if idx >= OUTPUT_SIZE:
            break  # No procesar más clases si ya hemos alcanzado el número de clases fijas
        class_mapping[class_folder] = idx
        class_path = os.path.join(dataset_path, class_folder)

        # Recorrer recursivamente todas las subcarpetas en la carpeta de clase
        for root, dirs, files in os.walk(class_path):
            for image_name in files:
                # Filtrar solo imágenes
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue  # Saltar archivos que no sean imágenes

                image_path = os.path.join(root, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"❌ ERROR: No se pudo leer '{image_path}'. Se omitirá.")
                    continue

                img = cv2.resize(img, (28, 28))  # Redimensionar
                img = cv2.bitwise_not(img)  # Invertir colores para mejor contraste
                images.append(img.flatten().astype(np.float32) / 255.0)  # Normalizar
                labels.append(idx)  # Etiqueta según la carpeta
                print(f"✔️ Imagen cargada: {image_name}")  # Imprimir cuando se carga una imagen

    if not images:
        raise ValueError("❌ ERROR: No se pudieron cargar imágenes.")

    images = np.array(images)
    labels = np.array(labels)

    # Dividir en entrenamiento (60%) y prueba (40%)
    split_idx = int(0.6 * len(images))  # 60% para entrenamiento
    indices = np.random.permutation(len(images))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = images[train_idx], labels[train_idx]
    X_test, y_test = images[test_idx], labels[test_idx]

    # Convertir etiquetas a one-hot encoding
    y_train_one_hot = np.eye(OUTPUT_SIZE)[y_train]
    y_test_one_hot = np.eye(OUTPUT_SIZE)[y_test]

    return (X_train, y_train_one_hot), (X_test, y_test_one_hot), class_mapping


# Definir la red neuronal MLP
class MLP:
    def __init__(self):
        self.W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2. / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2. / HIDDEN_SIZE)
        self.b2 = np.zeros(OUTPUT_SIZE)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        ex = np.exp(x - np.max(x))
        return ex / ex.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.softmax(self.z2)

    def backward(self, X, y, output):
        m = X.shape[0]
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        dz1 = np.dot(dz2, self.W2.T) * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled, y_shuffled = X_train[permutation], y_train[permutation]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                output = self.forward(X_batch)
                dW1, db1, dW2, db2 = self.backward(X_batch, y_batch, output)
                self.update_params(dW1, db1, dW2, db2, lr)

            val_output = self.forward(X_val)
            val_loss = -np.mean(np.log(val_output[np.arange(len(val_output)), np.argmax(y_val, axis=1)]))
            accuracy = np.mean(np.argmax(val_output, axis=1) == np.argmax(y_val, axis=1))
            print(f"Epoch {epoch+1}, Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

# Entrenar y guardar el modelo
def train_and_save_model(dataset_path):
    (X_train, y_train), (X_test, y_test), class_mapping = load_nist19(dataset_path)
    
    model = MLP()
    model.train(X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE, LEARNING_RATE)

    np.savez('NITST/nist19_model.npz', W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2, class_mapping=class_mapping)
    print("✅ Modelo guardado correctamente.")

# Predecir un carácter
def predict_char(image_path, model_path='NIST/nist19_model.npz'):
    model_data = np.load(model_path, allow_pickle=True)
    model = MLP()
    model.W1, model.b1, model.W2, model.b2 = model_data['W1'], model_data['b1'], model_data['W2'], model_data['b2']
    class_mapping = model_data['class_mapping'].item()
    label_to_class = {v: k for k, v in class_mapping.items()}

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo leer la imagen.")
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    processed_img = img.flatten().astype(np.float32) / 255.0

    output = model.forward(processed_img[np.newaxis, :])
    prediction = np.argmax(output)
    return label_to_class[prediction]  

# Uso del código
if __name__ == "__main__":

    start_time = time.time()  # Iniciar el contador

    #IMPORTANTE: Solo descomentar para entrenar al modelo

    #dataset_path = r"C:\Users\pc\Documents\UNEG_SEMESTRE_9\6- INTELIGENCIA ARTIFICIAL_2024\Proyecto#3.IA_Rojas_V_Landaeta_S._MLP\MLP-con-Backpropagation.-Sebastian-Landaeta-y-Victor-Rojas\EMNIST\by_class"
    #train_and_save_model(dataset_path)

    #print(f"✅ Entrenamiento completado.")

    test_image = "NIST/imagenes_de_prueba/F.png"
    print(f"El carácter reconocido es: {predict_char(test_image)}")
    
    end_time = time.time()  # Finalizar el contador
    elapsed_time = end_time - start_time

    print(f"Tiempo total: {elapsed_time:.4f} segundos.")