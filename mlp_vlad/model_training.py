import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

# --- КОНФИГУРАЦИЯ ---
CSV_FILE = "final_train_data.csv"
MODEL_SAVE_PATH = "badusb_model.pth"
SCALER_SAVE_PATH = "scaler_params.npy" # Нужно сохранить параметры скейлера для продакшена!

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
INPUT_SIZE = 14 # Количество признаков (F_Mean ... D_MinW)

# Настройка устройства (GPU если есть, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. ПОДГОТОВКА ДАННЫХ
# ==============================================================================
class BadUSBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # [batch, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data():
    print("Загрузка данных...")
    df = pd.read_csv(CSV_FILE)
    
    # Разделяем признаки и метки
    X = df.drop("Label", axis=1).values
    y = df["Label"].values

    # Разбивка: 80% Train, 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Разбивка Temp: 15% Val (это 0.75 от 20%), 5% Test (это 0.25 от 20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    print(f"Размеры выборок:")
    print(f"  Train: {len(X_train)} (80%)")
    print(f"  Val:   {len(X_val)} (15%)")
    print(f"  Test:  {len(X_test)} (5%)")

    # --- ВАЖНО: МАСШТАБИРОВАНИЕ (SCALING) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Обучаем скейлер только на Train
    X_val = scaler.transform(X_val)         # Применяем к Val
    X_test = scaler.transform(X_test)       # Применяем к Test
    
    # Сохраняем параметры скейлера (среднее и дисперсию), чтобы потом использовать в приложении
    np.save(SCALER_SAVE_PATH, [scaler.mean_, scaler.scale_])
    print("Параметры скейлера сохранены.")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ==============================================================================
# 2. МОДЕЛЬ (4 Layers, Fully Connected)
# ==============================================================================
class BadUSBClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BadUSBClassifier, self).__init__()
        # 4 слоя, сужающаяся архитектура
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1) # Бинарный выход

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Инициализация весов (Xavier Initialization)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# ==============================================================================
# 3. ФУНКЦИИ ОБУЧЕНИЯ
# ==============================================================================
def calculate_metrics(y_true, y_pred):
    # Округляем предсказания (0.1 -> 0, 0.9 -> 1)
    y_pred_tag = torch.round(y_pred)
    
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum / y_true.shape[0]
    
    # Переводим в CPU numpy для sklearn метрик
    y_t = y_true.cpu().detach().numpy()
    y_p = y_pred_tag.cpu().detach().numpy()
    f1 = f1_score(y_t, y_p, zero_division=0)
    
    return acc, f1

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        acc, f1 = calculate_metrics(y_batch, y_pred)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f1 += f1
        
    return epoch_loss / len(loader), epoch_acc / len(loader), epoch_f1 / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            acc, f1 = calculate_metrics(y_batch, y_pred)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1
            
    return epoch_loss / len(loader), epoch_acc / len(loader), epoch_f1 / len(loader)

# ==============================================================================
# 4. ВИЗУАЛИЗАЦИЯ
# ==============================================================================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

def plot_confusion_matrices(model, loaders):
    model.eval()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Train Set", "Validation Set", "Test Set"]
    
    with torch.no_grad():
        for i, (loader, title) in enumerate(zip(loaders, titles)):
            all_preds = []
            all_labels = []
            
            for X_b, y_b in loader:
                X_b = X_b.to(device)
                preds = model(X_b)
                all_preds.extend(torch.round(preds).cpu().numpy())
                all_labels.extend(y_b.numpy())
            
            cm = confusion_matrix(all_labels, all_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
            axes[i].set_title(title)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_xticklabels(['Human', 'Bot'])
            axes[i].set_yticklabels(['Human', 'Bot'])
            
            # Print report for Test set
            if title == "Test Set":
                print(f"\n--- Classification Report for {title} ---")
                print(classification_report(all_labels, all_preds, target_names=['Human', 'Bot']))

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 5. MAIN
# ==============================================================================
def main():
    # 1. Данные
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()
    
    train_dataset = BadUSBDataset(X_train, y_train)
    val_dataset = BadUSBDataset(X_val, y_val)
    test_dataset = BadUSBDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 2. Модель
    model = BadUSBClassifier(INPUT_SIZE).to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nНачало обучения...")
    best_val_loss = float('inf')
    
    # 3. Цикл обучения
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} (Val: {val_loss:.4f}) | "
              f"Acc: {train_acc:.4f} (Val: {val_acc:.4f}) | "
              f"F1: {train_f1:.4f} (Val: {val_f1:.4f})")
        
    print(f"\nМодель сохранена в: {MODEL_SAVE_PATH}")
    
    # 4. Графики
    plot_history(history)
    
    # 5. Матрицы ошибок
    # Загружаем лучшую версию модели перед тестом
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    plot_confusion_matrices(model, [train_loader, val_loader, test_loader])

if __name__ == "__main__":
    main()