import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_features_for_match(df, team_home, team_away):
    # Chọn các trận đấu liên quan đến hai đội
    match_df = df[((df['Đội Chủ Nhà'] == team_home) & (df['Đội Khách'] == team_away)) |
                  ((df['Đội Chủ Nhà'] == team_away) & (df['Đội Khách'] == team_home))]

    features = {}
    features['Đội Chủ Nhà'] = team_home
    features['Đội Khách'] = team_away
    # Phong độ gần đây
    features['Phong Độ Gần Đây Đội Nhà'] = match_df[match_df['Đội Chủ Nhà'] == team_home]['Kết quả'].rolling(3, min_periods=1).mean().iloc[-1] if len(match_df[match_df['Đội Chủ Nhà'] == team_home]) > 0 else 0
    features['Phong Độ Gần Đây Đội Khách'] = match_df[match_df['Đội Khách'] == team_away]['Kết quả'].rolling(3, min_periods=1).mean().iloc[-1] if len(match_df[match_df['Đội Khách'] == team_away]) > 0 else 0

    # Tỉ lệ thắng
    home_wins = df[(df['Đội Chủ Nhà'] == team_home) & (df['Kết quả'] == 0)].shape[0]
    home_total = df[df['Đội Chủ Nhà'] == team_home].shape[0]
    away_wins = df[(df['Đội Khách'] == team_away) & (df['Kết quả'] == 1)].shape[0]
    away_total = df[df['Đội Khách'] == team_away].shape[0]
    
    features['Tỉ Lệ Thắng Đội Nhà'] = home_wins / home_total if home_total > 0 else 0
    features['Tỉ Lệ Thắng Đội Khách'] = away_wins / away_total if away_total > 0 else 0

    # Tỉ lệ hòa
    home_draws = df[(df['Đội Chủ Nhà'] == team_home) & (df['Kết quả'] == 2)].shape[0]
    away_draws = df[(df['Đội Khách'] == team_away) & (df['Kết quả'] == 2)].shape[0]
    features['Tỉ Lệ Hòa Đội Nhà'] = home_draws / home_total if home_total > 0 else 0
    features['Tỉ Lệ Hòa Đội Khách'] = away_draws / away_total if away_total > 0 else 0

    # Bàn thắng trung bình
    features['Số Bàn Trung Bình Đội Nhà'] = df[df['Đội Chủ Nhà'] == team_home]['Bàn Thắng Đội Nhà'].mean()
    features['Số Bàn Trung Bình Đội Khách'] = df[df['Đội Khách'] == team_away]['Bàn Thắng Đội Khách'].mean()

    # Hiệu số bàn thắng
    features['Hiệu Số Bàn Thắng Đội Nhà'] = df[df['Đội Chủ Nhà'] == team_home]['Bàn Thắng Đội Nhà'].mean() - df[df['Đội Chủ Nhà'] == team_home]['Bàn Thắng Đội Khách'].mean()
    features['Hiệu Số Bàn Thắng Đội Khách'] = df[df['Đội Khách'] == team_away]['Bàn Thắng Đội Khách'].mean() - df[df['Đội Khách'] == team_away]['Bàn Thắng Đội Nhà'].mean()

    # Sạch lưới
    features['Sạch Lưới Đội Nhà'] = (df[df['Đội Chủ Nhà'] == team_home]['Bàn Thắng Đội Khách'] == 0).sum()
    features['Sạch Lưới Đội Khách'] = (df[df['Đội Khách'] == team_away]['Bàn Thắng Đội Nhà'] == 0).sum()

    # Thành tích đối đầu
    features['Thành Tích Đối Đầu Đội Nhà'] = df[(df['Đội Chủ Nhà'] == team_home) & (df['Đội Khách'] == team_away)]['Kết quả'].apply(lambda x: 1 if x == 0 else 0).sum()
    features['Thành Tích Đối Đầu Đội Khách'] = df[(df['Đội Khách'] == team_away) & (df['Đội Chủ Nhà'] == team_home)]['Kết quả'].apply(lambda x: 1 if x == 1 else 0).sum()

    # Chênh lệch bàn thắng
    features['Chênh Lệch Bàn Thắng Nhà-Khách'] = features['Hiệu Số Bàn Thắng Đội Nhà'] - features['Hiệu Số Bàn Thắng Đội Khách']
    features['Chênh Lệch Phong Độ Gần Đây'] = features['Phong Độ Gần Đây Đội Nhà'] - features['Phong Độ Gần Đây Đội Khách']

    # Chuỗi thắng gần đây
    features['Chuỗi Thắng Đội Nhà'] = df[df['Đội Chủ Nhà'] == team_home]['Kết quả'].eq(0).rolling(5, min_periods=1).sum().iloc[-1] if len(df[df['Đội Chủ Nhà'] == team_home]) > 0 else 0
    features['Chuỗi Thắng Đội Khách'] = df[df['Đội Khách'] == team_away]['Kết quả'].eq(1).rolling(5, min_periods=1).sum().iloc[-1] if len(df[df['Đội Khách'] == team_away]) > 0 else 0

    return features


def create_head_to_head_features(df, team_home, team_away):
    # Lọc ra các trận đối đầu giữa team_home và team_away, không phân biệt đội nhà hay đội khách
    head_to_head = df[((df['Đội Chủ Nhà'] == team_home) & (df['Đội Khách'] == team_away)) |
                      ((df['Đội Chủ Nhà'] == team_away) & (df['Đội Khách'] == team_home))]
    result = {}

    home_wins_1 = head_to_head[ (head_to_head['Đội Chủ Nhà'] == team_home) & (head_to_head['Kết quả'] == 0)].shape[0]
    home_wins_2 = head_to_head[ (head_to_head['Đội Khách'] == team_home) & (head_to_head['Kết quả'] == 1)].shape[0]
    home_wins_total = home_wins_1 + home_wins_2
    
    result['Tổng số trận thắng đội chủ nhà'] = home_wins_total

    away_wins_1 = head_to_head[ (head_to_head['Đội Chủ Nhà'] == team_away) & (head_to_head['Kết quả'] == 0)].shape[0]
    away_wins_2 = head_to_head[ (head_to_head['Đội Khách'] == team_away) & (head_to_head['Kết quả'] == 1)].shape[0]
    away_wins_total = away_wins_1 + away_wins_2
    
    result['Tổng số trận thắng đội khách'] = away_wins_total

    
    result['Tổng số trận hòa'] = head_to_head[head_to_head['Kết quả'] == 2].shape[0]
    return head_to_head, result
