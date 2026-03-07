import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# --- CONFIGURATION ---
DATASET_FILE = "rnn_dataset.pt"
MODEL_SAVE_PATH = "gru_model.pth"

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# GRU Parameters
INPUT_DIM = 2        # Dwell and Flight
HIDDEN_DIM = 64      # Size of network "memory"
NUM_LAYERS = 1       # Number of GRU layers (1-2 is enough)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. GRU MODEL ARCHITECTURE
# ==============================================================================
class GRUBotDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(GRUBotDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # batch_first=True means tensors have shape [Batch, Seq_Len, Features]
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape = [batch_size, seq_len, 2]
        
        # Initialize hidden state with zeros (optional, PyTorch does this automatically, but it's clearer)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # Pass data through GRU
        # out contains hidden states for ALL time steps
        out, _ = self.gru(x, h0)
        
        # WE ONLY NEED THE LAST STEP (when the network has seen all 20 events)
        # out[:, -1, :] takes all batches, last time step, all hidden features
        out = out[:, -1, :]
        
        # Pass through linear layer and sigmoid
        out = self.fc(out)
        return self.sigmoid(out)

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def calculate_metrics(y_true, y_pred):
    y_pred_tag = torch.round(y_pred)
    correct = (y_pred_tag == y_true).sum().float()
    acc = correct / y_true.shape[0]
    
    y_t = y_true.cpu().detach().numpy()
    y_p = y_pred_tag.cpu().detach().numpy()
    f1 = f1_score(y_t, y_p, zero_division=0)
    
    return acc.item(), f1

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss, epoch_acc, epoch_f1 = 0, 0, 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        acc, f1 = calculate_metrics(y_batch, y_pred)
        
        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_f1 += f1
        
    return epoch_loss / len(loader), epoch_acc / len(loader), epoch_f1 / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss, epoch_acc, epoch_f1 = 0, 0, 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            acc, f1 = calculate_metrics(y_batch, y_pred)
            
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_f1 += f1
            
    return epoch_loss / len(loader), epoch_acc / len(loader), epoch_f1 / len(loader)

# ==============================================================================
# 3. VISUALIZATION
# ==============================================================================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('GRU Loss History')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('GRU Accuracy History')
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
            sns.heatmap(cm, annot=True, fmt='g', cmap='Purples', ax=axes[i], cbar=False)
            axes[i].set_title(title)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_xticklabels(['Human', 'Bot'])
            axes[i].set_yticklabels(['Human', 'Bot'])
            
            if title == "Test Set":
                print(f"\n--- Classification Report ({title}) ---")
                print(classification_report(all_labels, all_preds, target_names=['Human', 'Bot']))

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    # 1. Load data
    print("Loading tensors...")
    try:
        data = torch.load(DATASET_FILE)
    except FileNotFoundError:
        print(f"File {DATASET_FILE} not found. First run the dataset preparation script.")
        return

    train_dataset = TensorDataset(data["X_train"], data["y_train"])
    val_dataset = TensorDataset(data["X_val"], data["y_val"])
    test_dataset = TensorDataset(data["X_test"], data["y_test"])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 2. Model initialization
    model = GRUBotDetector(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    
    print("\nStarting GRU training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} (Val: {val_loss:.4f}) | "
              f"Acc: {train_acc:.4f} (Val: {val_acc:.4f})")
        
    print(f"\n✅ Training completed. Best model saved to: {MODEL_SAVE_PATH}")
    
    # 3. Visualize results
    plot_history(history)
    
    # Load best weights for final testing
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    plot_confusion_matrices(model, [train_loader, val_loader, test_loader])

if __name__ == "__main__":
    main()