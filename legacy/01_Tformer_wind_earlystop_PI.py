#%%
import matplotlib.pyplot as plt
import copy
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

plt.switch_backend('TkAgg')  # Switch to TkAgg backend before creating figure

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     # random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.enabled = False
setup_seed(20)

# CUDA support 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Transformer functions definition
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, time_steps, d_model, nhead, num_layers, dim_feedforward, dropout, num_classes):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        
        # Project input to d_model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, time_steps, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        # Multi-layer structure for the output classifier
        self.fc = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, num_classes)
        )

    def _initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [batch_size, time_steps, input_dim]
        batch_size = x.size(0)
        
        # Project input to d_model dimension
        x = self.input_proj(x.float())  # [batch_size, time_steps, d_model]
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :self.time_steps, :]
        
        # Pass through Transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, time_steps, d_model] because batch_first=True
        
        # Use both global average pooling and global max pooling over the time dimension, then concatenate
        x_avg = x.mean(dim=1)         # [batch_size, d_model]
        x_max, _ = x.max(dim=1)       # [batch_size, d_model]
        x = torch.cat([x_avg, x_max], dim=1)  # [batch_size, 2*d_model]
        # Also update output layer's input dim accordingly (fc = nn.Linear(2*d_model, num_classes))
        
        output = self.fc(x)
        return output

def fit(epoch, model, trainloader, loss_fn, optimizer, scheduler, device):
    model.train()
    correct = 0
    total = 0
    running_loss = 0
    batch_count = 0
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to(device), y.long().to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_max = torch.argmax(y_pred, dim=1)
            correct += (y_pred_max == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

        batch_count += 1

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    for x, y in dataloader:
        if torch.cuda.is_available():
            x, y = x.to(device), y.long().to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        y_pred_max = torch.argmax(y_pred, dim=1)
        correct += (y_pred_max == y).sum().item()
        total += y.size(0)
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    avg_acc = correct / total
    return avg_loss, avg_acc

#%% Physics-informed feature engineering function
def add_physics_features(x_data):
    """
    Add physics-informed features to the input data.
    
    Assumes column order: origin_wind_speed, final_wind, peak_current, 
                          rotational_speed, Tambient, TIGBT_junction, Tcase
    
    Parameters:
    -----------
    x_data : numpy array
        Input features of shape (n_samples, 7)
    
    Returns:
    --------
    x_with_features : numpy array
        Features with physics-informed features added, shape (n_samples, 10)
    """
    epsilon = 1e-6  # Small constant to avoid division by zero
    k_calibration = 1.0  # Calibration constant for R_th_estimate
    
    # Extract relevant columns
    # Column indices: 0=origin_wind, 1=final_wind, 2=peak_current, 3=rotational_speed,
    #                 4=Tambient, 5=TIGBT_junction, 6=Tcase
    peak_current = x_data[:, 2]
    T_IGBT_junction = x_data[:, 5]
    T_case = x_data[:, 6]
    
    # 1. Junction-to-Case Temperature Difference
    delta_T_jc = T_IGBT_junction - T_case
    
    # 2. Normalized Temperature Differential
    delta_T_norm = delta_T_jc / (peak_current + epsilon)
    
    # 3. Instantaneous Thermal Resistance Estimate
    # R_th_estimate = (T_junction - T_case) / (I² × k + ε)
    I_squared = peak_current ** 2
    R_th_estimate = delta_T_jc / (I_squared * k_calibration + epsilon)
    
    # Concatenate original features with physics-informed features
    x_with_features = np.concatenate([
        x_data,  # Original 7 features
        delta_T_jc.reshape(-1, 1),      # Feature 8
        delta_T_norm.reshape(-1, 1),    # Feature 9
        R_th_estimate.reshape(-1, 1)    # Feature 10
    ], axis=1)
    
    return x_with_features

# %% data preprocessing
data_path = r"../data"
data_file = os.listdir(data_path)
data_file = [data_file[3], data_file[2], data_file[0], data_file[1]]
print("Processing files:", data_file)

window = 100
stride = 10

train_rate = 0.7
val_rate = 0.1

# 1. Load all files, split into train/val/test, but do NOT scale yet. Store splits temporarily.
all_x_train, all_y_train = [], []
all_x_val, all_y_val = [], []
all_x_test, all_y_test = [], []

for i, filename in enumerate(data_file):

    file_path = os.path.join(data_path, filename)
    print(f"Processing file {filename} with label {i}")
    try:
        raw_data = pd.read_excel(file_path)
        raw_data = raw_data.drop("time", axis=1)
        raw_data["label"] = i

        data = np.array(raw_data)

        x = data[:, :-1]      # features (7 original features)
        y = data[:, -1]        # label
        
        # Add physics-informed features
        x = add_physics_features(x)  # Now x has 10 features (7 original + 3 physics-informed)

        # Split into train/val/test
        total_len = len(x)
        idx_train = int(total_len * train_rate)
        idx_val = int(total_len * (train_rate + val_rate))

        x_train = x[:idx_train]
        y_train = y[:idx_train]
        x_val = x[idx_train:idx_val]
        y_val = y[idx_train:idx_val]
        x_test = x[idx_val:]
        y_test = y[idx_val:]

        all_x_train.append(x_train)
        all_y_train.append(y_train)
        all_x_val.append(x_val)
        all_y_val.append(y_val)
        all_x_test.append(x_test)
        all_y_test.append(y_test)

    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        continue

# 2. Combine all training data to fit MinMaxScaler
combined_x_train = np.vstack(all_x_train)
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(combined_x_train)

# 3. Normalize ALL sets using the fitted scaler, then windowize
def windowize(data_x, data_y, window, stride):
    win_x = []
    win_y = []
    for j in range(0, len(data_x) - window + 1, stride):
        data_t = data_x[j:j + window]
        win_x.append(data_t)
        win_y.append(data_y[j])
    return win_x, win_y

# 4. Clear lists for storage of final normalized, windowized samples
data_train_total = []
label_train_total = []
data_val_total = []
label_val_total = []
data_test_total = []
label_test_total = []

for i in range(len(data_file)):
    # Skip any files that did not have any data after filter
    if i >= len(all_x_train):
        continue
    # Normalize
    x_train_norm = min_max_scaler.transform(all_x_train[i])
    x_val_norm = min_max_scaler.transform(all_x_val[i])
    x_test_norm = min_max_scaler.transform(all_x_test[i])

    # Window-ize
    win_x_train, win_y_train = windowize(x_train_norm, all_y_train[i], window, stride)
    win_x_val, win_y_val = windowize(x_val_norm, all_y_val[i], window, stride)
    win_x_test, win_y_test = windowize(x_test_norm, all_y_test[i], window, stride)

    data_train_total.extend(win_x_train)
    label_train_total.extend(win_y_train)
    data_val_total.extend(win_x_val)
    label_val_total.extend(win_y_val)
    data_test_total.extend(win_x_test)
    label_test_total.extend(win_y_test)

#%% form dataset and dataloader

# Properly convert data and labels to tensors of the correct dtype and shape
train_data_tensor = torch.tensor(np.array(data_train_total), dtype=torch.float32)
train_label_tensor = torch.tensor(np.array(label_train_total), dtype=torch.long)
val_data_tensor = torch.tensor(np.array(data_val_total), dtype=torch.float32)
val_label_tensor = torch.tensor(np.array(label_val_total), dtype=torch.long)
test_data_tensor = torch.tensor(np.array(data_test_total), dtype=torch.float32)
test_label_tensor = torch.tensor(np.array(label_test_total), dtype=torch.long)

train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
val_dataset = TensorDataset(val_data_tensor, val_label_tensor)
test_dataset = TensorDataset(test_data_tensor, test_label_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)

#%% Transformer definition
input_dim = 10 # 7 original features + 3 physics-informed features
time_steps = window
d_model = 128
nhead = 4
num_layers = 4
dim_feedforward = 512
dropout = 0.01
num_classes = 4

model = TransformerClassifier(input_dim, time_steps, d_model, nhead, 
                              num_layers, dim_feedforward, dropout, 
                              num_classes)
model._initialize_parameters()

if torch.cuda.is_available():
    model.to('cuda') 

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.8
)
epochs = 500

# ------ Advanced Early Stopping Strategy ------
# Criteria:
#   - Monitor validation loss.
#   - If validation loss does not decrease by at least min_delta for "patience" epochs, stop.
#   - Optionally, if val_acc does not increase, can be considered.
patience = 30               # Number of epochs with no improvement after which training will be stopped
min_delta = 1e-4            # Minimum change to qualify as improvement
best_val_loss = float('inf')
best_val_acc = 0
cur_patience = 0
best_model_wts = None
best_epoch = 0

train_loss = []
train_acc = []
test_loss = []
test_acc = []
val_loss = []
val_acc = []

for epoch in range(epochs):

    # Train step
    epoch_loss, epoch_acc = fit(epoch, model, train_dataloader, loss_fn, optimizer, scheduler, device)
    scheduler.step()

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    # Validation step (for early stopping)
    epoch_val_loss, epoch_val_acc = evaluate(model, val_dataloader, loss_fn, device)
    val_loss.append(epoch_val_loss)
    val_acc.append(epoch_val_acc)

    # Test step (for tracking progress)
    epoch_test_loss, epoch_test_acc = evaluate(model, test_dataloader, loss_fn, device)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

    print('epoch: ', epoch,
          'loss: ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'val_loss: ', round(epoch_val_loss, 3),
          'val_accuracy:', round(epoch_val_acc, 3),
          'test_loss: ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )

    # Early stopping logic: Check validation loss (with min_delta)
    improvement = best_val_loss - epoch_val_loss
    if epoch > 100:
        if improvement > min_delta:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            cur_patience = 0
            print(f'Validation loss improved, model saved at epoch {epoch}')
        else:
            cur_patience += 1
            print(f'No improvement for {cur_patience} epochs.')
            if cur_patience >= patience:
                print('Early stopping triggered at epoch', epoch)
                break

# If training completed and best_model_wts is None (no improvement), save last weights
if best_model_wts is None:
    best_model_wts = copy.deepcopy(model.state_dict())

os.makedirs('./model_wind_pth_earlystop', exist_ok=True)
PATH = './model_wind_pth_earlystop/Transformer_PI.pth'   
torch.save(best_model_wts, PATH)   

print(f"Best model found at epoch {best_epoch} with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
# Save training history
history = {
    'train_loss': train_loss,
    'train_acc': train_acc, 
    'val_loss': val_loss,
    'val_acc': val_acc,
    'test_loss': test_loss,
    'test_acc': test_acc
}

# Create directory if it doesn't exist
os.makedirs('./training_history_wind_earlystop', exist_ok=True)

# Save as numpy arrays, include val_loss/val_acc as well
np.save('./training_history_wind_earlystop/train_loss_PI.npy', np.array(train_loss))
np.save('./training_history_wind_earlystop/train_acc_PI.npy', np.array(train_acc))
np.save('./training_history_wind_earlystop/val_loss_PI.npy', np.array(val_loss))
np.save('./training_history_wind_earlystop/val_acc_PI.npy', np.array(val_acc))
np.save('./training_history_wind_earlystop/test_loss_PI.npy', np.array(test_loss))
np.save('./training_history_wind_earlystop/test_acc_PI.npy', np.array(test_acc))


#%% show
plt.switch_backend('TkAgg')
plt.figure()
plt.plot(range(1, len(train_acc) + 1), train_acc, label="train_acc")
plt.plot(range(1, len(test_acc) + 1), test_acc, label="test_acc")
plt.plot(range(1, len(val_acc) + 1), val_acc, label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy curves - Transformer with Physics-Informed Features")
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(range(1, len(train_loss) + 1), train_loss, label="train_loss")
plt.plot(range(1, len(test_loss) + 1), test_loss, label="test_loss")
plt.plot(range(1, len(val_loss) + 1), val_loss, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curves - Transformer with Physics-Informed Features")
plt.legend()
plt.show(block=False)# %%

#%% Compare the correct and predicted labels on validation and test set

def get_predictions_and_labels(dataloader, model, device):
    model.eval()
    correct_labels = []
    predicted_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct_labels.extend(y.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
    return np.array(correct_labels), np.array(predicted_labels)

model.load_state_dict(torch.load('./model_wind_pth_earlystop/Transformer_PI.pth', weights_only=True))

val_correct_labels, val_predicted_labels = get_predictions_and_labels(val_dataloader, model, device)
test_correct_labels, test_predicted_labels = get_predictions_and_labels(test_dataloader, model, device)

def print_metrics(true_labels, pred_labels, dataset_name='Dataset'):
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    rec = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    print(f"\n{dataset_name} Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")

# Compute and print metrics for validation set
print_metrics(val_correct_labels, val_predicted_labels, dataset_name='Validation')

# Compute and print metrics for test set
print_metrics(test_correct_labels, test_predicted_labels, dataset_name='Test')

# Plot confusion matrix for validation set
plt.figure(figsize=(6, 5))
cm_val = confusion_matrix(val_correct_labels, val_predicted_labels)
ConfusionMatrixDisplay(confusion_matrix=cm_val).plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Validation Set (Transformer with PI Features)')
plt.show(block=False)  

# Plot confusion matrix for test set
plt.figure(figsize=(6, 5))
cm_test = confusion_matrix(test_correct_labels, test_predicted_labels)
ConfusionMatrixDisplay(confusion_matrix=cm_test).plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Test Set (Transformer with PI Features)')
plt.show(block=False)

# %%
