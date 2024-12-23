import pandas as pd
import glob

# './data' 아래 모든 CSV 파일 경로 가져오기
file_paths = glob.glob('./training_data/**/*.csv', recursive=True)

# 빈 리스트를 생성하여 데이터를 담을 준비
all_data = []

# 각 파일을 읽고 필요한 전처리 수행 후 리스트에 추가
for file_path in file_paths:
    df = pd.read_csv(file_path)
    
    # 필요 없는 열 제거
    df = df.drop(columns=["ghosts_count", "player_direction"], errors="ignore")    
    all_data.append(df)

# 모든 데이터를 하나의 DataFrame으로 합치기
merged_data = pd.concat(all_data, ignore_index=True)

# num_lives 열에 대한 통계 계산
column_name = 'difficulty'
if column_name in merged_data.columns:
    mean_value = merged_data[column_name].mean()
    min_value = merged_data[column_name].min()
    max_value = merged_data[column_name].max()
    
    print(f"Column: {column_name}")
    print(f"Mean: {mean_value}")
    print(f"Min: {min_value}")
    print(f"Max: {max_value}")

# 최종 데이터 확인
print(merged_data.columns.tolist())
