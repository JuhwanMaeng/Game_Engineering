import gymnasium as gym
from atariari.benchmark.wrapper import AtariARIWrapper
import os
import csv
import ale_py

# 경로 설정
input_folder = './dataset/grandchallenge'
output_folder = './dataset/converted_challenge'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Gym 환경 초기화
gym.register_envs(ale_py)
env = AtariARIWrapper(gym.make('MsPacmanDeterministic-v0', render_mode=None, full_action_space=True, frameskip=1, repeat_action_probability=0.0))

# 점수 변환 설정
max_player_score = 144  # 순환 발생 최대 값
score_conversion_factor = 10 / 16  # 16 증가마다 10점으로 변환

def save_to_csv(info, frame_number, prev_score, total_score, filename):
    current_score = int(info['labels']['player_score'])

    # 누적 점수 계산
    if current_score < prev_score:  # 순환 발생
        delta = (max_player_score - prev_score) + current_score
    else:  # 정상 증가
        delta = current_score - prev_score

    # 점수 변환
    total_score += delta
    converted_score = int(total_score * score_conversion_factor)

    data = {
        'frame_number': frame_number,
        'enemy_sue_x': info['labels']['enemy_sue_x'],
        'enemy_inky_x': info['labels']['enemy_inky_x'],
        'enemy_pinky_x': info['labels']['enemy_pinky_x'],
        'enemy_blinky_x': info['labels']['enemy_blinky_x'],
        'enemy_sue_y': info['labels']['enemy_sue_y'],
        'enemy_inky_y': info['labels']['enemy_inky_y'],
        'enemy_pinky_y': info['labels']['enemy_pinky_y'],
        'enemy_blinky_y': info['labels']['enemy_blinky_y'],
        'player_x': info['labels']['player_x'],
        'player_y': info['labels']['player_y'],
        'fruit_x': info['labels']['fruit_x'],
        'fruit_y': info['labels']['fruit_y'],
        'ghosts_count': info['labels']['ghosts_count'],
        'player_direction': info['labels']['player_direction'],
        'dots_eaten_count': info['labels']['dots_eaten_count'],
        'player_score': converted_score,
        'num_lives': info['labels']['num_lives']
    }

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if file.tell() == 0:  # 파일에 데이터가 없다면 헤더 작성
            writer.writeheader()
        writer.writerow(data)

    # 이전 점수와 누적 점수 반환
    return current_score, total_score

# txt 파일 처리
for txt_file in os.listdir(input_folder):
    if not txt_file.endswith('.txt'):
        continue

    input_path = os.path.join(input_folder, txt_file)
    output_path = os.path.join(output_folder, txt_file.replace('.txt', '.csv'))

    # 환경 리셋 및 초기화
    env.reset()
    prev_score = 0
    total_score = 0
    frame_number = 0

    # txt 파일에서 행동 로드
    with open(input_path, 'r') as file:
        lines = file.readlines()
        actions = [int(line.split(',')[-1].strip()) for line in lines[2:]]  # 첫 줄은 헤더

    # 시뮬레이션 실행
    for action in actions:
        observation, reward, done, info = env.step(action)
        prev_score, total_score = save_to_csv(info, frame_number, prev_score, total_score, output_path)
        frame_number += 1  # 프레임 번호 증가

        if done:  # 게임 종료 처리
            print(f"Simulation for {txt_file} ended at frame {frame_number}.")
            break

    print(f"Simulation for {txt_file} completed. Output saved to {output_path}")

env.close()
print(f"Simulation complete. All converted files are saved to {output_folder}")
