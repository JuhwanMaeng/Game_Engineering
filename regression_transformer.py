import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
FRAME_STEP = 30
BATCH_SIZE = 200
HIDDEN_LAYER_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.01
INPUT_SIZE = 13 
OUTPUT_SIZE = 1
DATA_PATH = 'training_data'

class NoamOpt:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step_update(self):
        self.step_num += 1
        lr = self.d_model ** -0.5 * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_heads=8, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.embedding = nn.Linear(input_size, hidden_layer_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, FRAME_STEP, hidden_layer_size))
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_layer_size, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_layer_size)
        self.fc1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.fc3 = nn.Linear(hidden_layer_size // 2, output_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = x.transpose(0, 1)
        
        for layer in self.attention_layers:
            x = layer(x)

        x = self.norm(x)
        x = x[-1, :, :]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class DataLoaderWithNormalization:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size

    def load_data(self):
        file_paths = glob.glob(f'{self.data_path}/**/*.csv', recursive=True)
        all_data = []

        for file_path in tqdm(file_paths, desc="Loading CSV files"):
            data = pd.read_csv(file_path)
            data = data.drop(columns=["frame_number", "ghosts_count", "fruit_x", "fruit_y", "difficulty"], errors="ignore")
            data = data.iloc[270:]  # 데이터를 앞부분에서 270번째 이후로 자른다고 가정
            
            if len(data) < FRAME_STEP:  # 데이터가 FRAME_STEP보다 짧으면 스킵
                continue

            player_score = data['player_score'].values
            features = data.drop(columns=['player_score']).values

            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            X, y = self.create_sequences(features, player_score)

            if X.shape[0] > 0 and y.shape[0] > 0:  # 유효 데이터만 추가
                all_data.append((X, y))

        X_all = np.concatenate([x for x, _ in all_data], axis=0)
        y_all = np.concatenate([y for _, y in all_data], axis=0)

        X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2)
        
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                torch.tensor(y_train, dtype=torch.float32)),
                                batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                            torch.tensor(y_val, dtype=torch.float32)),
                                batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    @staticmethod
    def create_sequences(data, player_scores, frame_steps=FRAME_STEP):
        X, y = [], []
        if len(data) < frame_steps:  # 데이터가 너무 짧으면 빈 배열 반환
            return np.empty((0, frame_steps, data.shape[1])), np.empty((0, 1))

        target = player_scores[-1]  # 해당 파일의 마지막 player_score 값
        for i in range(0, len(data) - frame_steps):
            X.append(data[i:i + frame_steps])  # X는 30개의 연속된 프레임
            y.append(target)  # y는 항상 마지막 player_score 값

        return np.array(X), np.array(y).reshape(-1, 1)


def run_epoch(model, dataloader, optimizer, criterion, scheduler, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0
    dataloader = tqdm(dataloader, desc="Training" if is_train else "Validation")  # tqdm으로 진행 상태 표시
    for X, y in dataloader:
        X, y = X.to(device), y.to(device).squeeze(-1)  # y 크기를 (배치 크기,)로 변경
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y)  # 출력도 (배치 크기,)로 사용
        if is_train:
            loss.backward()
            scheduler.step_update()
        total_loss += loss.item()

        # 현재 진행 중인 손실 업데이트 (tqdm 표시용)
        dataloader.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

if __name__ == "__main__":
    data_loader = DataLoaderWithNormalization(DATA_PATH, BATCH_SIZE)
    train_loader, val_loader = data_loader.load_data()

    model = TimeSeriesTransformer(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = NoamOpt(optimizer, d_model=HIDDEN_LAYER_SIZE)

    criterion = nn.MSELoss()
    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, scheduler, is_train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, scheduler, is_train=False)
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
