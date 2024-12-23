import os
import csv

# 경로 설정
input_folder = './dataset/raw_playdata'
output_folder = './dataset/converted_playdata'
max_player_score = 144  # 순환 발생 최대값
score_conversion_factor = 10 / 16  # 16 증가마다 10점으로 변환

# 출력 폴더 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def fix_and_convert_player_score(input_file, output_file, max_score, conversion_factor):
    """player_score를 변환하고 새로운 파일로 저장하는 함수"""
    with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        prev_score = 0
        total_score = 0

        for row in reader:
            current_score = int(row['player_score'])

            # 순환 점수 계산
            if current_score < prev_score:  # 순환 발생
                total_score += (max_score - prev_score) + current_score
            else:  # 정상 증가
                total_score += (current_score - prev_score)

            # 수정된 점수 계산 (16 증가마다 10으로 변환)
            converted_score = int(total_score * conversion_factor)

            # 수정된 점수 반영
            row['player_score'] = converted_score
            writer.writerow(row)

            # 이전 점수를 현재 점수로 업데이트
            prev_score = current_score

# 모든 CSV 파일 처리
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):  # CSV 파일만 처리
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        
        # 점수 변환
        fix_and_convert_player_score(input_file, output_file, max_player_score, score_conversion_factor)
        print(f"Processed: {filename} -> {output_file}")

print("모든 파일 변환 완료!")