import gymnasium as gym
from atariari.benchmark.wrapper import AtariARIWrapper
import os
import csv
import numpy as np
import pygame
from pygame.locals import QUIT
import ale_py

# 경로 설정
input_folder = './dataset/grandchallenge'
output_folder = './dataset/converted_challenge'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Gym 환경 초기화
gym.register_envs(ale_py)
env = AtariARIWrapper(gym.make('MsPacmanNoFrameskip-v4', render_mode="rgb_array", full_action_space=True,  frameskip=1,repeat_action_probability=0.0 ))

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Ms. Pacman Replay")
clock = pygame.time.Clock()

# 점수 변환 설정
max_player_score = 144  # 순환 발생 최대 값
score_conversion_factor = 10 / 16  # 16 증가마다 10점으로 변환

prev_score = 0
total_score = 0

def save_to_csv(info, filename):
    global prev_score, total_score

    current_score = int(info['labels']['player_score'])

    # 누적 점수 계산
    if current_score < prev_score:  # 순환 발생
        delta = (max_player_score - prev_score) + current_score
    else:  # 정상 증가
        delta = current_score - prev_score

    # 큰 변화량 처리 (변환된 점수 누적)
    total_score += delta

    # 점수 변환 (16 증가마다 10으로 변환)
    converted_score = int(total_score * score_conversion_factor)

    data = {
        'frame_number': info['frame_number'],
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
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(data)

    # 이전 점수 업데이트
    prev_score = current_score

# txt 파일 처리
for txt_file in os.listdir(input_folder):
    if not txt_file.endswith('.txt'):
        continue

    input_path = os.path.join(input_folder, txt_file)
    output_path = os.path.join(output_folder, txt_file.replace('.txt', '.csv'))

    # 환경 리셋
    env.reset()
    done = False

    # txt 파일에서 행동 로드
    with open(input_path, 'r') as file:
        lines = file.readlines()
        actions = [int(line.split(',')[-1].strip()) for line in lines[2:]]  # 두 번째 줄부터 데이터
        terminals = [line.split(',')[3].strip() == 'True' for line in lines[2:]]

    # 시뮬레이션 실행
    expected_frames = len(actions)
    for frame, (action, terminal) in enumerate(zip(actions, terminals)):
        if done:
            break

        observation, reward, done, info = env.step(action)

        # 실제 종료 여부 확인
        if terminal and not done:
            print(f"Mismatch detected in {txt_file}: Terminal state expected but simulation continued at frame {frame + 1}.")
        elif not terminal and done:
            print(f"Mismatch detected in {txt_file}: Simulation terminated unexpectedly at frame {frame + 1}.")

        save_to_csv(info, output_path)

        # 화면 렌더링
        frame_array = env.render()
        pygame_frame = pygame.surfarray.make_surface(frame_array)
        rotated_frame = pygame.transform.rotate(pygame_frame, -90)
        flipped_frame = pygame.transform.flip(rotated_frame, True, False)
        screen.blit(pygame.transform.scale(flipped_frame, (640, 480)), (0, 0))
        pygame.display.flip()
        
        clock.tick(60)

        # 종료 이벤트 확인
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
                break

    # 시뮬레이션 종료 시 조건 확인
    if frame + 1 < expected_frames:
        print(f"Simulation for {txt_file} ended earlier than expected at frame {frame + 1}/{expected_frames}.")
    elif not done:
        print(f"Simulation for {txt_file} survived beyond expected frames ({expected_frames}).")
    else:
        print(f"Simulation for {txt_file} completed exactly as expected ({expected_frames} frames).")

pygame.quit()
env.close()
print(f"Simulation complete. Converted files saved to {output_folder}")
