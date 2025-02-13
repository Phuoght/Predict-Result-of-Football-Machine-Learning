import requests
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_data():
     api_key = '218cb831cdb04fa88cc1efe39ae40673'

     # url giải đẩu Premier League
     url = 'https://api.football-data.org/v4/competitions/PL/matches'

     # Cài đặt header để xác thực yêu cầu
     headers = {
     'X-Auth-Token': api_key
     }
     # Dữ liệu trận đấu từ 2015 tới 28-05-2023
     #matches_df = pd.read_csv("Model_Train/football_matches_dataset.csv") # Dùng cho app.py
     matches_df = pd.read_csv("football_matches_dataset.csv") # Dùng cho main.ipynb
     matches_df = matches_df[matches_df['league'] == 'EPL']
     matches_df = matches_df.iloc[:, 0:8]
     matches_df = matches_df[matches_df['id'] > 1022 ]
     matches_df['Kết quả']= np.where(matches_df['home_score'] > matches_df['away_score'],
          '0', np.where(matches_df['home_score'] < matches_df['away_score'],'1', '2'))
     string_date = np.asanyarray(matches_df['date'].str.split(' '))
     for i in range(len(matches_df)):
          matches_df.iloc[i,3] = string_date[i][2] + " " + string_date[i][0] + " " 
          + string_date[i][1]
     matches_df['date'] = pd.to_datetime(matches_df['date']).dt.date
     matches_df = matches_df.rename(columns={'home_team': "Đội Chủ Nhà",
      'away_team': 'Đội Khách', 'season': 'Mùa Giải', 'date': 'Ngày Diễn Ra',
     "home_score": "Bàn Thắng Đội Nhà", "away_score": "Bàn Thắng Đội Khách"})

     matches_df = matches_df.drop(['id', 'league', 'Mùa Giải'], axis=1)


     # Tạo một danh sách để chứa dữ liệu các trận đấu
     all_matches = []

     # Lấy dữ liệu cho từng mùa giải từ 2023 đến nay
     for season_year in range(2023, datetime.now().year + 1):
          params = {
               'season': str(season_year) # Lấy dữ liệu theo mùa giải
          }
          # Gửi yêu cầu HTTP GET
          response = requests.get(url, headers=headers, params=params)
          # Kiểm tra nếu yêu cầu thành công
          if response.status_code == 200:
               data = response.json()
               matches = data['matches']
               # Thêm các trận đấu vào danh sách all_matches
               all_matches.extend(matches)

     # Chuyển dữ liệu thành DataFrame
     df = pd.json_normalize(all_matches)

     columns_to_keep = ['utcDate', 'homeTeam.shortName', 'awayTeam.shortName',
                    'score.fullTime.home', 'score.fullTime.away', 'score.winner']
     df_cleaned = df[columns_to_keep]

     # Đổi tên các cột
     df_cleaned.columns = ['Date', 'Home Team', 'Away Team', 
                           'Home Score', 'Away Score', 'Winner']

     # Chuyển đổi các cột kiểu dữ liệu
     df_cleaned.loc[:, 'Date'] = pd.to_datetime(df_cleaned['Date']).dt.date
     df_cleaned.loc[:, 'Home Score'] = pd.to_numeric(df_cleaned['Home Score'],errors='coerce')
     df_cleaned.loc[:, 'Away Score'] = pd.to_numeric(df_cleaned['Away Score'],errors='coerce')

     # Xử lý dữ liệu null
     df_cleaned = df_cleaned.dropna()

     # Chuẩn hóa kết quả
     df_cleaned['Winner'] = np.where(df_cleaned['Winner'] == "HOME_TEAM", '0', 
                         np.where(df_cleaned['Winner'] == "AWAY_TEAM", '1', '2'))

     data = {'Đội Chủ Nhà': df_cleaned['Home Team'],
              'Đội Khách': df_cleaned['Away Team'],
              "Ngày Diễn Ra": df_cleaned['Date'],
               "Bàn Thắng Đội Nhà":df_cleaned['Home Score'] ,
               "Bàn Thắng Đội Khách": df_cleaned['Away Score'],
                 "Kết quả": df_cleaned['Winner']}
     data = pd.DataFrame(data)

     # Ghép 2 data lại
     matches_summary = pd.concat([matches_df, data], axis=0, ignore_index=True)
     matches_summary['Kết quả'] = matches_summary['Kết quả'].astype(int)

     # Từ điển chuẩn hóa tên đội
     team_corrections = {
          'Manchester United': ['Man United', 'Manchester United'],
          'Manchester City': ['Man City', 'Manchester City'],
          'Newcastle United': ['Newcastle', 'Newcastle United'],
          'Wolverhampton Wanderers': ['Wolverhampton', 'Wolverhampton Wanderers'],
          'Brighton': ['Brighton', 'Brighton Hove'],
          'Sheffield United': ['Sheffield United', 'Sheffield Utd'],
          'Nottingham Forest': ['Nottingham', 'Nottingham Forest'],
          'Leicester': ['Leicester', 'Leicester City']
          }

# Hàm chuẩn hóa tên đội bóng
     def normalize_team_name(team_name):
          for standard_name, variations in team_corrections.items():
               if team_name in variations:
                    return standard_name
          return team_name
     
     matches_summary['Đội Chủ Nhà'] = matches_summary['Đội Chủ Nhà'].apply(normalize_team_name)
     matches_summary['Đội Khách']=matches_summary['Đội Khách'].apply(normalize_team_name)

     return matches_summary   