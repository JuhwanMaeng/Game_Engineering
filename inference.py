import gymnasium as gym
from atariari.benchmark.wrapper import AtariARIWrapper
import pygame
from pygame.locals import QUIT, KEYDOWN, K_UP, K_DOWN, K_LEFT, K_RIGHT, KEYUP
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import ale_py

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

# Load LSTM model
def load_model(model_path, input_size, hidden_layer_size, output_size):
    model = LSTMModel(input_size, hidden_layer_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Create sequences for LSTM input
def create_sequences(data, frame_steps, stride= 1):
    X = []
    for i in range(0, len(data) - frame_steps, stride):
        X.append(data[i:i + frame_steps])
    return np.array(X)

# Predict difficulty using the model
def predict(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch in data_loader:
            X_batch = X_batch[0]  # Unpack from tuple
            if X_batch.size(0) == 0:  # Check if batch is empty
                print("Empty batch encountered in DataLoader.")
                continue
            y_pred = model(X_batch)
            predictions.append(y_pred.argmax(dim=1).cpu().numpy())
    if not predictions:  # Check if predictions list is empty
        print("No predictions were made.")
        return np.array([])  # Return empty array to avoid ValueError
    return np.concatenate(predictions)

# Main integration
if __name__ == "__main__":
    # Hyperparameters
    INPUT_SIZE = 15
    HIDDEN_LAYER_SIZE = 64
    OUTPUT_SIZE = 5
    FRAME_STEP = 30
    BATCH_SIZE = 10
    MODEL_PATH = "0.6914_best_model.pth"

    # Load LSTM model
    model = load_model(MODEL_PATH, INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)

    # Initialize game
    gym.register_envs(ale_py)
    env = AtariARIWrapper(gym.make('MsPacmanNoFrameskip-v4', render_mode="rgb_array", full_action_space=True, frameskip=1, repeat_action_probability=0.0))
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Ms. Pacman")

    key_to_action = {
        K_UP: 2,    # Up
        K_DOWN: 5,  # Down
        K_LEFT: 4,  # Left
        K_RIGHT: 3, # Right
    }

    env.reset()
    done = False
    clock = pygame.time.Clock()
    action = 0

    # Store gameplay frames for prediction
    frames = []
    prev_score = 0
    total_score = 0
    while not done:
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
            elif event.type == KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
            elif event.type == KEYUP:
                pass

        observation, reward, done, info = env.step(action)
        screen.fill((0, 0, 0))
        frame = pygame.surfarray.make_surface(env.render())
        rotated_frame = pygame.transform.rotate(frame, -90)
        flipped_frame = pygame.transform.flip(rotated_frame, True, False)
        screen.blit(pygame.transform.scale(flipped_frame, (640, 480)), (0, 0))
        pygame.display.flip()

        current_score = int(info['labels']['player_score'])
        # 순환 점수 계산
        if current_score < prev_score:
            total_score += (144 - prev_score) + current_score
        else:
            total_score += (current_score - prev_score)

        # 수정된 점수 계산 (16 증가마다 10으로 변환)
        converted_score = int(total_score * (10 / 16))

        # Extract features for prediction
        features = [
            info['labels']['enemy_sue_x'], info['labels']['enemy_inky_x'],
            info['labels']['enemy_pinky_x'], info['labels']['enemy_blinky_x'],
            info['labels']['enemy_sue_y'], info['labels']['enemy_inky_y'],
            info['labels']['enemy_pinky_y'], info['labels']['enemy_blinky_y'],
            info['labels']['player_x'], info['labels']['player_y'], 
            info['labels']['fruit_x'], info['labels']['fruit_y'], info['labels']['dots_eaten_count'],
            converted_score, info['labels']['num_lives']]

        frames.append(features)
        prev_score = current_score
        if len(frames) >= FRAME_STEP:
            sequence = create_sequences(np.array(frames), FRAME_STEP)
            if sequence.shape[0] == 0:
                print("No valid sequences generated for prediction.")
                continue  # Skip to the next frame
            test_loader = DataLoader(TensorDataset(torch.tensor(sequence, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
            predictions = predict(model, test_loader)
            if predictions.size == 0:  # Check if predictions array is empty
                print("No difficulty predictions available.")
            else:
                print(f"예상 난이도: {predictions[-1]}")
            frames.pop(0)  # Remove the oldest frame
            
            if predictions[-1] == 0 : 
                env.unwrapped.ale.setRAM(19,-1)
            elif predictions[-1] == 1 : 
                env.unwrapped.ale.setRAM(19,0)
            elif predictions[-1] == 2 : 
                env.unwrapped.ale.setRAM(19,1)
            elif predictions[-1] == 3 : 
                env.unwrapped.ale.setRAM(19,2)    
            elif predictions[-1] == 4 : 
                env.unwrapped.ale.setRAM(19,3)    

        clock.tick(60)  # 60 FPS

    env.close()
    pygame.quit()