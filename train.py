import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

# Hyperparameters
FRAME_STEP = 10
BATCH_SIZE = 200
HIDDEN_LAYER_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01
INPUT_SIZE = 15
OUTPUT_SIZE = 5
DATA_PATH = 'training_data'

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, num_layers=2)
        self.fc1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.fc3 = nn.Linear(hidden_layer_size // 2, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc1(hn[-1])
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def create_sequences(data, labels, frame_steps):
    X, y = [], []
    for i in range(len(data) - frame_steps):
        X.append(data[i:i + frame_steps]) 
        y.append(labels[i + frame_steps])
    return np.array(X), np.array(y)


# Data Loader
class DifficultyDataLoader:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size

    def load_data(self):
        file_paths = glob.glob(f'{self.data_path}/**/*.csv', recursive=True)
        all_data = []

        for file_path in tqdm(file_paths, desc="Loading CSV files"):
            data = pd.read_csv(file_path)
            data = data.drop(columns=["ghosts_count", "player_direction"], errors="ignore")   
            data = data.iloc[270:]
            all_data.append(data)

        combined_data = pd.concat(all_data, ignore_index=True)

        features = combined_data.drop(columns=['frame_number', 'difficulty']).values
        labels = combined_data['difficulty'].values

        # 시계열 데이터 생성
        X, y = create_sequences(features, labels, FRAME_STEP)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.long)),  # dtype=torch.long
                          batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                            torch.tensor(y_val, dtype=torch.long)),  # dtype=torch.long
                                batch_size=self.batch_size, shuffle=False)


        return train_loader, val_loader


# Training Function
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation Function
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Main Execution
if __name__ == "__main__":
    # Load data
    data_loader = DifficultyDataLoader(DATA_PATH, BATCH_SIZE)
    train_loader, val_loader = data_loader.load_data()

    # Initialize model
    model = LSTMModel(input_size=INPUT_SIZE, hidden_layer_size=HIDDEN_LAYER_SIZE, output_size=OUTPUT_SIZE)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("New best model, saving...")
            torch.save(model.state_dict(), 'best_model.pth')