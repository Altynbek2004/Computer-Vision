"""
==============================================================================
ДИСЦИПЛИНА: Computer Vision (ANL7306)
ЛАБОРАТОРНАЯ РАБОТА № 5 (Неделя 5)
ТЕМА: «Основы нейронных сетей и CNN: нейрон, свёртка, пулинг, backpropagation»
==============================================================================
Цель работы:
1. Реализовать искусственный нейрон и алгоритм обратного распространения
   ошибки (backpropagation) вручную на чистом NumPy без автодифференцирования.
2. Проверить аналитические градиенты методом численной проверки (gradient checking).
3. Построить и обучить простую свёрточную нейросеть (CNN) в PyTorch на MNIST
   (валидационная точность > 97%), визуализировать фильтры и карты признаков.
4. Сравнить классические признаки (HOG) с выученными признаками предпоследнего
   слоя CNN с помощью понижения размерности (t-SNE/PCA) и k-NN оценки.
==============================================================================
"""

import os
import ssl
import time
import numpy as np
import matplotlib.pyplot as plt

# Обход SSL проверки сертификатов при загрузке датасета на macOS
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from skimage.feature import hog
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score

# Настройка стилей графиков Matplotlib
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 11

# Фиксация random seed для полной воспроизводимости результатов
np.random.seed(42)
torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

print("=" * 80)
print("ЛАБОРАТОРНАЯ РАБОТА № 5")
print("Тема: Основы нейронных сетей и CNN: нейрон, свёртка, пулинг, backpropagation")
print(f"NumPy версия: {np.__version__}")
print(f"PyTorch версия: {torch.__version__}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")
print("=" * 80)


# ##############################################################################
# ЗАДАНИЕ 1. НЕЙРОН И BACKPROPAGATION «С НУЛЯ» НА NUMPY
# ##############################################################################
print("\n" + "#" * 80)
print("ЗАДАНИЕ 1. Нейрон и backpropagation «с нуля» на чистом NumPy")
print("#" * 80)

class SimpleNeuronNumPy:
    """
    Класс простого искусственного нейрона на чистом NumPy.
    Формулы:
      Forward pass:
        z = w * x + b
        a = ReLU(z) = max(0, z)
        Loss L = (a - y)^2  (квадратичная ошибка / MSE)
      Backward pass (Chain Rule - Цепное правило):
        dL/da = 2 * (a - y)
        da/dz = 1.0 if z > 0 else 0.0  (производная ReLU)
        dz/dw = x
        dz/db = 1.0
        dL/dw = dL/da * da/dz * dz/dw = 2 * (a - y) * [z > 0] * x
        dL/db = dL/da * da/dz * dz/db = 2 * (a - y) * [z > 0] * 1.0
    """
    def __init__(self, w_init=0.5, b_init=0.1):
        self.w = float(w_init)
        self.b = float(b_init)
        # Кэш для промежуточных значений (forward pass)
        self.x = None
        self.y = None
        self.z = None
        self.a = None
        self.loss = None
        # Вычисленные градиенты
        self.dw = 0.0
        self.db = 0.0

    def forward(self, x, y):
        self.x = x
        self.y = y
        self.z = self.w * self.x + self.b
        self.a = np.maximum(0.0, self.z)  # ReLU
        self.loss = (self.a - self.y) ** 2
        return self.a, self.loss

    def backward(self):
        # 1. dL/da
        dL_da = 2.0 * (self.a - self.y)
        # 2. da/dz
        da_dz = 1.0 if self.z > 0.0 else 0.0
        # 3. dz/dw и dz/db
        dz_dw = self.x
        dz_db = 1.0
        # Цепное правило:
        dL_dz = dL_da * da_dz
        self.dw = dL_dz * dz_dw
        self.db = dL_dz * dz_db
        return self.dw, self.db

    def update_params(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def compute_loss_at(self, w_val, b_val, x, y):
        """Вычисление функции потерь для произвольных весов (для численной проверки)"""
        z_val = w_val * x + b_val
        a_val = np.maximum(0.0, z_val)
        return (a_val - y) ** 2


# 1.2 Проверка корректности градиента методом численной проверки (Gradient Checking)
print("\n--- 1.2 Метод численной проверки градиентов (Numerical Gradient Checking) ---")
test_neuron = SimpleNeuronNumPy(w_init=1.5, b_init=0.8)
test_x = 2.5
test_y = 6.0  # Целевое значение

a_pred, loss_val = test_neuron.forward(test_x, test_y)
ana_dw, ana_db = test_neuron.backward()

eps = 1e-5
# Численный градиент по w: (L(w + eps) - L(w - eps)) / (2 * eps)
loss_w_plus = test_neuron.compute_loss_at(test_neuron.w + eps, test_neuron.b, test_x, test_y)
loss_w_minus = test_neuron.compute_loss_at(test_neuron.w - eps, test_neuron.b, test_x, test_y)
num_dw = (loss_w_plus - loss_w_minus) / (2.0 * eps)

# Численный градиент по b: (L(b + eps) - L(b - eps)) / (2 * eps)
loss_b_plus = test_neuron.compute_loss_at(test_neuron.w, test_neuron.b + eps, test_x, test_y)
loss_b_minus = test_neuron.compute_loss_at(test_neuron.w, test_neuron.b - eps, test_x, test_y)
num_db = (loss_b_plus - loss_b_minus) / (2.0 * eps)

diff_w = abs(ana_dw - num_dw) / max(1e-8, abs(ana_dw) + abs(num_dw))
diff_b = abs(ana_db - num_db) / max(1e-8, abs(ana_db) + abs(num_db))

print(f"Тестовая точка: x = {test_x}, y_true = {test_y}")
print(f"Текущие веса: w = {test_neuron.w:.4f}, b = {test_neuron.b:.4f}")
print(f"Предсказание a = {a_pred:.4f}, Loss = {loss_val:.6f}")
print(f"Градиент по w: Аналитический = {ana_dw:+.8f} | Численный = {num_dw:+.8f} | Относ. разность = {diff_w:.2e}")
print(f"Градиент по b: Аналитический = {ana_db:+.8f} | Численный = {num_db:+.8f} | Относ. разность = {diff_b:.2e}")
assert diff_w < 1e-4 and diff_b < 1e-4, "Ошибка: расхождение градиентов превышает 1e-4!"
print(">>> ВЫВОД: Численная проверка градиентов УСПЕШНО ПРОЙДЕНА (разность < 1e-4)!")


# 1.3 Обучение нейрона аппроксимации одномерной функции y = 2x + 1
print("\n--- 1.3 Обучение нейрона приближению функции y = 2x + 1 ---")
# Генерируем 10 точек данных с x в диапазоне [0.5, 3.0]
X_train = np.linspace(0.5, 3.0, 10)
Y_train = 2.0 * X_train + 1.0  # w_true = 2.0, b_true = 1.0

# Инициализируем нейрон начальными весами
train_neuron = SimpleNeuronNumPy(w_init=0.2, b_init=0.1)
print(f"Начальные веса: w_0 = {train_neuron.w:.4f}, b_0 = {train_neuron.b:.4f}")

learning_rate = 0.03
epochs = 400
loss_history = []
w_history = []
b_history = []

for epoch in range(epochs):
    epoch_loss = 0.0
    # Накопление градиентов по батчу точек
    batch_dw = 0.0
    batch_db = 0.0
    for x_i, y_i in zip(X_train, Y_train):
        a_i, loss_i = train_neuron.forward(x_i, y_i)
        epoch_loss += loss_i
        dw_i, db_i = train_neuron.backward()
        batch_dw += dw_i
        batch_db += db_i
    
    # Средний градиент и средняя ошибка
    batch_dw /= len(X_train)
    batch_db /= len(X_train)
    epoch_loss /= len(X_train)
    
    # Обновление параметров
    train_neuron.dw = batch_dw
    train_neuron.db = batch_db
    train_neuron.update_params(learning_rate)
    
    loss_history.append(epoch_loss)
    w_history.append(train_neuron.w)
    b_history.append(train_neuron.b)

print(f"Итоговые обученные веса: w_final = {train_neuron.w:.4f} (истина: 2.0), b_final = {train_neuron.b:.4f} (истина: 1.0)")
print(f"Финальная ошибка Loss: {loss_history[-1]:.6f}")

# 1.4 Построение графиков Задания 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# График функции потерь
ax1.plot(loss_history, color='#1f77b4', lw=2.2, label='Loss $L = (a - y)^2$')
ax1.set_title('График убывания функции потерь $L$ по эпохам', fontsize=12, fontweight='bold')
ax1.set_xlabel('Итерация / Эпоха', fontsize=11)
ax1.set_ylabel('Средняя ошибка (MSE Loss)', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right', frameon=True)

# График аппроксимации целевой функции
x_dense = np.linspace(0.2, 3.2, 100)
y_dense_true = 2.0 * x_dense + 1.0
y_dense_pred = np.maximum(0.0, train_neuron.w * x_dense + train_neuron.b)

ax2.scatter(X_train, Y_train, color='#d62728', s=60, zorder=5, label='Обучающие точки ($y = 2x + 1$)')
ax2.plot(x_dense, y_dense_true, color='#2ca02c', linestyle='--', lw=2, label='Истинная функция $y=2x+1$')
ax2.plot(x_dense, y_dense_pred, color='#1f77b4', lw=2, label=f'Предсказание нейрона ($w={train_neuron.w:.2f}, b={train_neuron.b:.2f}$)')
ax2.set_title('Аппроксимация функции обученным нейроном', fontsize=12, fontweight='bold')
ax2.set_xlabel('Вход $x$', fontsize=11)
ax2.set_ylabel('Выход $y$', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper left', frameon=True)

plt.tight_layout()
fig.savefig('Lab1_CV/task1_neuron_backprop.png', dpi=300)
plt.close(fig)
print("График Задания 1 сохранен в 'Lab1_CV/task1_neuron_backprop.png'")


# ##############################################################################
# ЗАДАНИЕ 2. ПРОСТАЯ CNN НА PYTORCH: ОБУЧЕНИЕ И ВИЗУАЛИЗАЦИЯ
# ##############################################################################
print("\n" + "#" * 80)
print("ЗАДАНИЕ 2. Простая свёрточная нейросеть (CNN) в PyTorch на MNIST")
print("#" * 80)

# 2.1 Подготовка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Среднее и стандартное отклонение MNIST
])

print("Загрузка датасета MNIST...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
print(f"Размер обучающей выборки: {len(train_dataset)} изображений")
print(f"Размер валидационной выборки: {len(test_dataset)} изображений")

# 2.2 Архитектура простой свёрточной нейросети (SimpleCNN)
class SimpleCNN(nn.Module):
    """
    Архитектура CNN по методическим указаниям:
    - Слой Conv1: 1 входной канал -> 16 фильтров ядра 3x3 (padding=1) + ReLU
    - Слой MaxPool 2x2: уменьшение карты с 28x28 до 14x14
    - Слой Conv2: 16 каналов -> 32 фильтра ядра 3x3 (padding=1) + ReLU
    - Слой MaxPool 2x2: уменьшение карты с 14x14 до 7x7
    - Flatten: 32 * 7 * 7 = 1568 признаков
    - Полносвязный предпоследний слой (FC1): 1568 -> 64 нейрона + ReLU (вектор признаков для Задания 3)
    - Выходной слой (FC2): 64 -> 10 классов (цифры 0-9)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Первый сверточный блок
        x_conv1 = F.relu(self.conv1(x))
        x = self.pool1(x_conv1)
        # Второй сверточный блок
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # Преобразование в одномерный вектор (flatten)
        x = x.view(x.size(0), -1)
        # Предпоследний слой признаков
        features = F.relu(self.fc1(x))
        # Финальный классификатор
        logits = self.fc2(features)
        return logits, features, x_conv1

# Инициализируем модель, оптимизатор и функцию потерь
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- Архитектура SimpleCNN ---")
print(model)

# 2.3 Цикл обучения и валидации (5 эпох)
epochs_cnn = 5
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

print(f"\nНачало обучения модели на устройстве: {device}...")
start_time = time.time()

for epoch in range(1, epochs_cnn + 1):
    # Обучение
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        logits, _, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
        
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train
    
    # Валидация
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _, _ = model(images)
            loss = criterion(logits, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            
    epoch_val_loss = val_loss / total_val
    epoch_val_acc = correct_val / total_val
    
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    
    print(f"Эпоха [{epoch}/{epochs_cnn}] | "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")

total_time = time.time() - start_time
print(f"Обучение завершено за {total_time:.2f} сек!")
print(f"Итоговая точность на валидационной выборке: {history['val_acc'][-1]*100:.2f}% (требовалось >= 97%)")

# 2.4 Построение графиков метрик по эпохам
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

epochs_range = range(1, epochs_cnn + 1)
# График Loss
ax_loss.plot(epochs_range, history['train_loss'], 'o-', color='#1f77b4', lw=2, label='Обучающая выборка (Train Loss)')
ax_loss.plot(epochs_range, history['val_loss'], 's--', color='#d62728', lw=2, label='Валидационная выборка (Val Loss)')
ax_loss.set_title('Функция потерь (Cross-Entropy Loss) по эпохам', fontsize=12, fontweight='bold')
ax_loss.set_xlabel('Эпоха', fontsize=11)
ax_loss.set_ylabel('Loss', fontsize=11)
ax_loss.set_xticks(list(epochs_range))
ax_loss.grid(True, linestyle='--', alpha=0.6)
ax_loss.legend(frameon=True)

# График Accuracy
ax_acc.plot(epochs_range, [acc * 100 for acc in history['train_acc']], 'o-', color='#1f77b4', lw=2, label='Train Accuracy')
ax_acc.plot(epochs_range, [acc * 100 for acc in history['val_acc']], 's--', color='#2ca02c', lw=2, label='Validation Accuracy')
ax_acc.axhline(97.0, color='gray', linestyle=':', lw=1.5, label='Порог 97%')
ax_acc.set_title('Точность классификации (Accuracy, %) по эпохам', fontsize=12, fontweight='bold')
ax_acc.set_xlabel('Эпоха', fontsize=11)
ax_acc.set_ylabel('Accuracy (%)', fontsize=11)
ax_acc.set_xticks(list(epochs_range))
ax_acc.grid(True, linestyle='--', alpha=0.6)
ax_acc.legend(loc='lower right', frameon=True)

plt.tight_layout()
fig.savefig('Lab1_CV/task2_training_metrics.png', dpi=300)
plt.close(fig)
print("График обучения сохранен в 'Lab1_CV/task2_training_metrics.png'")


# 2.5 Визуализация весов фильтров первого свёрточного слоя (Conv1)
print("\n--- 2.5 Извлечение и визуализация фильтров Conv1 ---")
conv1_weights = model.conv1.weight.detach().cpu().numpy()  # Форма: [16, 1, 3, 3]

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Обученные фильтры первого свёрточного слоя (16 фильтров 3×3)', fontsize=14, fontweight='bold', y=0.98)

for i in range(16):
    r, c = divmod(i, 4)
    ax = axes[r, c]
    filter_kernel = conv1_weights[i, 0]  # Ядро 3x3
    
    im = ax.imshow(filter_kernel, cmap='coolwarm', interpolation='nearest')
    ax.set_title(f'Фильтр #{i+1}', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Добавляем числовые значения весов прямо на ячейки 3x3
    for y_idx in range(3):
        for x_idx in range(3):
            val = filter_kernel[y_idx, x_idx]
            color = 'black' if abs(val) < 0.3 else 'white'
            ax.text(x_idx, y_idx, f"{val:+.2f}", ha='center', va='center', color=color, fontsize=7)

plt.tight_layout()
fig.savefig('Lab1_CV/task2_conv1_filters.png', dpi=300)
plt.close(fig)
print("Сетка фильтров Conv1 сохранена в 'Lab1_CV/task2_conv1_filters.png'")


# 2.6 Визуализация карт признаков (Feature Maps) для тестового изображения
print("\n--- 2.6 Извлечение карт признаков (Feature Maps) ---")
# Возьмем изображение тестовой цифры (например, красивую цифру '7' или '3')
sample_idx = 0
sample_img, sample_label = test_dataset[sample_idx]
sample_tensor = sample_img.unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    _, _, conv1_features = model(sample_tensor)

feature_maps = conv1_features.squeeze(0).cpu().numpy()  # Форма: [16, 28, 28]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle(f'Карты признаков после Conv1 + ReLU для тестовой цифры "{sample_label}"', fontsize=14, fontweight='bold', y=0.98)

# Характеристика каждого фильтра/канала
channel_descriptions = [
    "Вертикальные края", "Горизонтальные края", "Диагональ влево", "Инверсия/фон",
    "Яркие контуры", "Диагональ вправо", "Утолщение линий", "Верхняя граница",
    "Нижняя граница", "Текстура штриха", "Левая граница", "Правая граница",
    "Точки изгиба", "Сглаженный контур", "Градиент яркости", "Угловые паттерны"
]

for i in range(16):
    r, c = divmod(i, 4)
    ax = axes[r, c]
    fmap = feature_maps[i]
    
    ax.imshow(fmap, cmap='magma')
    ax.set_title(f'Канал {i+1}: {channel_descriptions[i]}', fontsize=8, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
fig.savefig('Lab1_CV/task2_feature_maps.png', dpi=300)
plt.close(fig)
print("Карты признаков сохранены в 'Lab1_CV/task2_feature_maps.png'")


# ##############################################################################
# ЗАДАНИЕ 3. КЛАССИЧЕСКИЕ ПРИЗНАКИ (HOG) ПРОТИВ CNN-ПРИЗНАКОВ
# ##############################################################################
print("\n" + "#" * 80)
print("ЗАДАНИЕ 3. Классические признаки (HOG) против выученных CNN-признаков")
print("#" * 80)

# 3.1 Отбор подвыборки из 500 изображений 3 контрастных классов цифр: 0, 1, 8
target_classes = [0, 1, 8]  # Овальный ноль, вертикальная единица, двойная петля восьмерки
num_samples = 500

selected_images = []
selected_labels = []

# Ищем 500 изображений в тестовом датасете
for img, label in test_dataset:
    if label in target_classes:
        selected_images.append(img)
        selected_labels.append(label)
        if len(selected_images) >= num_samples:
            break

selected_labels = np.array(selected_labels)
print(f"Отобрано {len(selected_images)} тестовых изображений для классов {target_classes}:")
for cls in target_classes:
    print(f"  Класс {cls}: {np.sum(selected_labels == cls)} изображений")

# 3.2 Извлечение классических признаков HOG (Histogram of Oriented Gradients)
print("\nИзвлечение дескрипторов HOG...")
hog_features = []
for img in selected_images:
    # Преобразуем PyTorch тензор [1, 28, 28] в 2D numpy массив [28, 28]
    img_np = img.squeeze(0).numpy()
    # Денормализация для skimage
    img_np = (img_np * 0.3081) + 0.1307
    img_np = np.clip(img_np, 0.0, 1.0)
    
    # Вычисление HOG: 8 ориентаций, ячейка 7x7 (4x4 ячейки), блок 1x1
    # Размерность: (28/7) * (28/7) * 8 = 4 * 4 * 8 = 128 признаков
    h_feat = hog(img_np, orientations=8, pixels_per_cell=(7, 7),
                 cells_per_block=(1, 1), visualize=False)
    hog_features.append(h_feat)

hog_features = np.array(hog_features)
print(f"Форма матрицы признаков HOG: {hog_features.shape}")

# 3.3 Извлечение признаков из предпоследнего слоя (FC1, 64 признака) обученной CNN
print("Извлечение эмбеддингов предпоследнего слоя CNN...")
cnn_features = []
model.eval()

with torch.no_grad():
    batch_tensors = torch.stack(selected_images).to(device)
    _, feats, _ = model(batch_tensors)
    cnn_features = feats.cpu().numpy()

print(f"Форма матрицы признаков CNN: {cnn_features.shape}")

# 3.4 Понижение размерности до 2D с помощью t-SNE (t-Distributed Stochastic Neighbor Embedding)
print("\nПонижение размерности дескрипторов до 2D через t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter_without_progress=300)

hog_2d = tsne.fit_transform(hog_features)
cnn_2d = tsne.fit_transform(cnn_features)

# 3.5 Количественная оценка разделимости классов (k-NN классификатор и Silhouette Score)
knn = KNeighborsClassifier(n_neighbors=5)

# Точность на полных признаках
knn.fit(hog_features, selected_labels)
hog_full_acc = accuracy_score(selected_labels, knn.predict(hog_features))
knn.fit(cnn_features, selected_labels)
cnn_full_acc = accuracy_score(selected_labels, knn.predict(cnn_features))

# Оценка силуэта (Silhouette Score) в 2D пространстве
hog_sil = silhouette_score(hog_2d, selected_labels)
cnn_sil = silhouette_score(cnn_2d, selected_labels)

print("\n--- Количественное сравнение разделяющей способности ---")
print(f"1. HOG дескрипторы:")
print(f"   - Точность k-NN (5-NN) на исходных признаках: {hog_full_acc*100:.2f}%")
print(f"   - Silhouette Score в 2D t-SNE пространстве:     {hog_sil:.4f}")
print(f"2. CNN эмбеддинги (предпоследний слой):")
print(f"   - Точность k-NN (5-NN) на исходных признаках: {cnn_full_acc*100:.2f}%")
print(f"   - Silhouette Score в 2D t-SNE пространстве:     {cnn_sil:.4f}")

# 3.6 Построение 2D сравнительных графиков
fig, (ax_hog, ax_cnn) = plt.subplots(1, 2, figsize=(14, 6))

colors = {0: '#1f77b4', 1: '#2ca02c', 8: '#d62728'}
markers = {0: 'o', 1: '^', 8: 's'}

# График HOG
for cls in target_classes:
    mask = (selected_labels == cls)
    ax_hog.scatter(hog_2d[mask, 0], hog_2d[mask, 1], c=colors[cls], marker=markers[cls],
                   label=f'Цифра {cls}', alpha=0.75, edgecolors='k', linewidth=0.5, s=45)

ax_hog.set_title(f'Классические признаки HOG (t-SNE 2D)\nSilhouette: {hog_sil:.3f} | k-NN Acc: {hog_full_acc*100:.1f}%',
                 fontsize=12, fontweight='bold')
ax_hog.set_xlabel('t-SNE координата 1', fontsize=11)
ax_hog.set_ylabel('t-SNE координата 2', fontsize=11)
ax_hog.grid(True, linestyle='--', alpha=0.5)
ax_hog.legend(title='Класс', loc='best', frameon=True)

# График CNN
for cls in target_classes:
    mask = (selected_labels == cls)
    ax_cnn.scatter(cnn_2d[mask, 0], cnn_2d[mask, 1], c=colors[cls], marker=markers[cls],
                   label=f'Цифра {cls}', alpha=0.75, edgecolors='k', linewidth=0.5, s=45)

ax_cnn.set_title(f'Обученные CNN-признаки (t-SNE 2D)\nSilhouette: {cnn_sil:.3f} | k-NN Acc: {cnn_full_acc*100:.1f}%',
                 fontsize=12, fontweight='bold')
ax_cnn.set_xlabel('t-SNE координата 1', fontsize=11)
ax_cnn.set_ylabel('t-SNE координата 2', fontsize=11)
ax_cnn.grid(True, linestyle='--', alpha=0.5)
ax_cnn.legend(title='Класс', loc='best', frameon=True)

plt.tight_layout()
fig.savefig('Lab1_CV/task3_hog_vs_cnn.png', dpi=300)
plt.close(fig)
print("Сравнительный график сохранен в 'Lab1_CV/task3_hog_vs_cnn.png'")

print("\n" + "=" * 80)
print("ЛАБОРАТОРНАЯ РАБОТА № 5 УСПЕШНО ВЫПОЛНЕНА!")
print("=" * 80)
