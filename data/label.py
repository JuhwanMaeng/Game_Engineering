import os
import pandas as pd
import glob

# './data' 아래 모든 CSV 파일 경로 가져오기
file_paths = glob.glob('./data/**/*.csv', recursive=True)

# 수동으로 구간 설정
bins = [-1.14, 228.0, 456.0, 684.0, 912.0, 20000.0]
labels = [0, 1, 2, 3, 4]  # 각 구간에 매칭되는 레이블
# 최종 데이터 저장 경로 설정
output_dir = './final_data'
os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리 생성

# 각 파일 처리
for file_path in file_paths:
    df = pd.read_csv(file_path)
    
    if not df.empty:
        # 파일의 마지막 행의 player_score 값을 가져오기
        last_score = df['player_score'].iloc[-1]

        # 디버깅: 마지막 점수 출력
        print(f"Processing file: {file_path}, Last score: {last_score}")


        difficulty = pd.cut(
            [last_score],  # 마지막 점수 하나만 입력
            bins=bins,  # 수동으로 설정한 구간 사용
            labels=labels,  # 각 구간에 대한 레이블
            include_lowest=True  # 첫 구간에 포함
        )[0]
        # 디버깅: difficulty 값 출력
        print(f"Difficulty for {file_path}: {difficulty}")

        # 전체 파일에 동일한 difficulty 값 적용
        df['difficulty'] = difficulty
    
    # 새로운 경로 생성 및 저장
    # 파일 경로 유지: ./final_data/ 이하 동일 구조
    relative_path = os.path.relpath(file_path, './data')  # 상대 경로
    save_path = os.path.join(output_dir, relative_path)  # 저장 경로
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 중간 디렉토리 생성
    df.to_csv(save_path, index=False)

print(f"All files processed and saved to '{output_dir}'")
