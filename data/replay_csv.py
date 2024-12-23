import gymnasium as gym
from atariari.benchmark.wrapper import AtariARIWrapper
import os
import csv
import numpy as np
import ale_py

# 경로 설정
input_folder = './dataset/raw_playdata'

# Gym 환경 초기화
gym.register_envs(ale_py)
env = AtariARIWrapper(gym.make('MsPacmanNoFrameskip-v4', render_mode="human", full_action_space=False, frameskip=1, repeat_action_probability=0.0))

# CSV 파일 처리
for csv_file in os.listdir(input_folder):
    if not csv_file.endswith('.csv'):
        continue

    input_path = os.path.join(input_folder, csv_file)
    # 환경 리셋
    env.reset()
    done = False

    # CSV 파일에서 데이터 로드
    with open(input_path, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # 시뮬레이션 실행
    for row in rows:
        if done:
            break
        x = int(row['player_x'])
        y = int(row['player_y'])

        # RAM 데이터 수정
        env.unwrapped.ale.setRAM(10,x)
        env.unwrapped.ale.setRAM(16,y)
        # 행동 수행 (예: 0번 행동)
        observation, reward, done, info = env.step(0)
    print(f"Simulation for {csv_file} completed.")

env.close()
print(f"All simulations completed.")
