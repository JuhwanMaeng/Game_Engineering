import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
FRAME_STEP = 15
BATCH_SIZE = 200
HIDDEN_LAYER_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.01
INPUT_SIZE = 14
OUTPUT_SIZE = 5
DATA_PATH = 'training_data'

# Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_heads=8, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        
        # Embedding layer
        self.embedding = nn.Linear(input_size, hidden_layer_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, FRAME_STEP, hidden_layer_size))
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_layer_size,
            nhead=num_heads,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Feed-forward layers
        self.fc1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.fc3 = nn.Linear(hidden_layer_size // 2, output_size)

    def forward(self, x):
        # Add positional encoding
        x = self.embedding(x) + self.positional_encoding
        
        # Transformer expects input of shape [seq_len, batch_size, feature_size], so we transpose
        x = x.transpose(0, 1)
        
        # Pass through the transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the output of the last time step
        x = x[-1, :, :]
        
        # Feed-forward layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

def create_sequences(data, labels, frame_steps, stride=1):
    X, y = [], []
    for i in range(0, len(data) - frame_steps, stride):
        X.append(data[i:i + frame_steps])
        y.append(labels[i + frame_steps])
    return np.array(X), np.array(y)


# Data Loader with normalization
class DifficultyDataLoader:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size

    def load_data(self):
        file_paths = glob.glob(f'{self.data_path}/**/*.csv', recursive=True)
        all_data = []

        # Read all files and normalize the data
        for file_path in tqdm(file_paths, desc="Loading CSV files"):
            data = pd.read_csv(file_path)
            data = data.drop(columns=["ghosts_count", "fruit_x", "fruit_y"], errors="ignore")
            data = data.iloc[270:]
            all_data.append(data)

        combined_data = pd.concat(all_data, ignore_index=True)

        features = combined_data.drop(columns=['frame_number', 'difficulty']).values
        labels = combined_data['difficulty'].values

        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Create sequences for time series data
        X, y = create_sequences(features, labels, FRAME_STEP)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        # Create DataLoader for training and validation
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                torch.tensor(y_train, dtype=torch.long)),
                                  batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                              torch.tensor(y_val, dtype=torch.long)),
                                batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

# Label smoothing function
def label_smoothed_nll_loss(lprobs, target, eps):
    nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1))
    nll_loss = nll_loss.squeeze(-1)
    nll_loss = nll_loss.mean()
    smooth_loss = -lprobs.mean(dim=-1)
    smooth_loss = smooth_loss.mean()
    loss = (1.0 - eps) * nll_loss + eps * smooth_loss
    return loss

# Optimizer with NoamOpt learning rate scheduler
class NoamOpt:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.learning_rate = 0
        self.step = 0

    def step_update(self):
        self.step += 1
        self.learning_rate = self.d_model ** -0.5 * min(self.step ** -0.5, self.step * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def get_lr(self):
        return self.learning_rate

# Training and validation functions
def run_epoch(model, train_loader, val_loader, optimizer, criterion, scheduler, epoch, max_epoch=100):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        lprobs = nn.functional.log_softmax(output, dim=-1)
        loss = label_smoothed_nll_loss(lprobs, target, eps=0.1)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Validation after each epoch
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            lprobs = nn.functional.log_softmax(output, dim=-1)
            loss = label_smoothed_nll_loss(lprobs, target, eps=0.1)
            val_loss += loss.item()

    # Learning rate update based on validation loss
    scheduler.step(val_loss)

    print(f"Epoch [{epoch}/{max_epoch}] - Train Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")

    return total_loss / len(train_loader), val_loss / len(val_loader)


# Main Execution
if __name__ == "__main__":
    # Load data
    data_loader = DifficultyDataLoader(DATA_PATH, BATCH_SIZE)
    train_loader, val_loader = data_loader.load_data()

    # Initialize model
    model = TransformerModel(input_size=INPUT_SIZE, hidden_layer_size=HIDDEN_LAYER_SIZE, output_size=OUTPUT_SIZE).to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = NoamOpt(optimizer, d_model=HIDDEN_LAYER_SIZE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS):
        train_loss, val_loss = run_epoch(model, train_loader, val_loader, optimizer, criterion, scheduler, epoch)
        print(f"Epoch {epoch+1}: Train Loss {train_loss}, Validation Loss {val_loss}")