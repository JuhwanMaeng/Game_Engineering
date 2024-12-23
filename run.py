import gymnasium as gym
from atariari.benchmark.wrapper import AtariARIWrapper
import pygame, ale_py
from pygame.locals import QUIT, KEYDOWN, K_UP, K_DOWN, K_LEFT, K_RIGHT, KEYUP
import numpy as np
import csv
import os

# Game Init
gym.register_envs(ale_py)
env = AtariARIWrapper(gym.make('MsPacmanNoFrameskip-v4', render_mode="rgb_array", full_action_space=True,  frameskip=1, repeat_action_probability=0.0))
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Ms. Pacman")
key_to_action = {
    K_UP: 2,    # 위
    K_DOWN: 5,  # 아래
    K_LEFT: 4,  # 오른쪽
    K_RIGHT: 3, # 왼쪽
}
env.reset()
done = False
clock = pygame.time.Clock()
action = 0

save_folder = './dataset/playdata'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Player score transformation logic
max_player_score = 144  # 순환 발생 최대 값
score_conversion_factor = 10 / 16  # 16 증가마다 10점으로 변환

prev_score = 0
total_score = 0

def get_new_log_filename():
    num = 1
    while os.path.exists(os.path.join(save_folder, f'game_log_{num}.csv')):
        num += 1
    return os.path.join(save_folder, f'game_log_{num}.csv')

def save_to_csv(info, filename):
    global prev_score, total_score
    current_score = int(info['labels']['player_score'])

    if current_score < prev_score:
        delta = (max_player_score - prev_score) + current_score
    else: 
        delta = current_score - prev_score

    total_score += delta
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
    prev_score = current_score
log_filename = get_new_log_filename()

while not done:
    for event in pygame.event.get():
        if event.type == QUIT:
            done = True
        elif event.type == KEYDOWN:
            if event.key in key_to_action:
                action = key_to_action[event.key]
            if event.key == K_UP:
                print("K_UP")
            elif event.key == K_DOWN:
                print("K_DOWN")
            elif event.key == K_LEFT:
                print("K_LEFT")
            elif event.key == K_RIGHT:
                print("K_RIGHT")
        elif event.type == KEYUP:
            pass
    
    observation, reward, done, info = env.step(action)
    save_to_csv(info, log_filename)
    screen.fill((0, 0, 0))
    frame = pygame.surfarray.make_surface(env.render())
    rotated_frame = pygame.transform.rotate(frame, -90)
    flipped_frame = pygame.transform.flip(rotated_frame, True, False)
    screen.blit(pygame.transform.scale(flipped_frame, (640, 480)), (0, 0))
    pygame.display.flip()
    clock.tick(60) # 초당 60fps

env.close()
pygame.quit()

"""
0 : 위
2 : 아래
3 : 왼쪽
1 : 오른쪽
"""