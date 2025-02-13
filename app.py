from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import df_module as dm
from Model_Train import get_data_regulary as gdr
import joblib
import numpy as np
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':

        # Lấy tên 2 đội từ Form người dùng nhập
        team_home = request.form['choose_home']
        team_away = request.form['choose_away']
        
        # Lấy tên file từ đường dẫn đầy đủ để lấy tên đội bóng
        team_home = os.path.splitext(os.path.basename(team_home))[0]
        team_away = os.path.splitext(os.path.basename(team_away))[0]

        # Tải dataframe
        #df = gdr.fetch_data()
        df = pd.read_csv("Model_Train/data_to_4-11-2024.csv")

        # Gọi hàm tạo đặc trưng cho 2 đội
        features = dm.create_features_for_match(df, team_home, team_away)

        # Phong độ gần đây của cả 2 dành cho hiện lên web cho người dùng xem    
        five_tail_match_team_home = df[(df['Đội Chủ Nhà'] == team_home) | (df['Đội Khách'] == team_home)].tail(5)
        five_tail_match_team_away = df[(df['Đội Chủ Nhà'] == team_away) | (df['Đội Khách'] == team_away)].tail(5)

        # Đếm số lần xuất hiện của từng kết quả trong 5 trận gần nhất
        home_last_5_match_1 = five_tail_match_team_home[five_tail_match_team_home['Đội Chủ Nhà'] == team_home]['Kết quả'].value_counts()
        home_last_5_match_2 = five_tail_match_team_home[five_tail_match_team_home['Đội Khách']== team_home]['Kết quả'].value_counts()
        home_wins = home_last_5_match_1.get(0,0) + home_last_5_match_2.get(1,0)
        home_loses = home_last_5_match_1.get(1,0) + home_last_5_match_2.get(0,0)
        home_draws = home_last_5_match_1.get(2,0) + home_last_5_match_2.get(2,0)
        
        away_last_5_match_1 = five_tail_match_team_away[five_tail_match_team_away['Đội Chủ Nhà']== team_away]['Kết quả'].value_counts()
        away_last_5_match_2 = five_tail_match_team_away[five_tail_match_team_away['Đội Khách'] == team_away]['Kết quả'].value_counts()
        away_wins = away_last_5_match_1.get(0,0) + away_last_5_match_2.get(1,0)
        away_loses = away_last_5_match_1.get(1,0) + away_last_5_match_2.get(0,0)
        away_draws = away_last_5_match_1.get(2,0) + away_last_5_match_2.get(2,0)

        
        # Lịch sử đối đầu giữa hai đội
        sum_match, head_to_head_stats = dm.create_head_to_head_features(df, team_home, team_away)


        # Tạo các đặc trưng cho 2 đội bóng
        X_test_df = pd.DataFrame([features])
        le = LabelEncoder()
        all_teams = list(X_test_df['Đội Chủ Nhà']) + list(X_test_df['Đội Khách'])
        le.fit(all_teams)
        X_test_df['Đội Chủ Nhà'] = le.transform(X_test_df['Đội Chủ Nhà'])
        X_test_df['Đội Khách'] = le.transform(X_test_df['Đội Khách'])
        X_test = X_test_df.to_numpy()
        
        #
        # Load mô hình
        stacking_model = joblib.load('stacking_model.joblib')
        y_prob = stacking_model.predict_proba(X_test)

        # Đặt ngưỡng tối thiểu cho lớp hòa
        threshold = 0.5

        # # Nếu xác suất lớp hòa (lớp 2) lớn hơn ngưỡng, bỏ qua lớp hòa và chọn lớp thắng hoặc thua
        if (y_prob[0][2] > threshold) and (sum_match.shape[0] < 15):
            # Nếu xác suất hòa quá cao, chọn lớp có xác suất cao nhất ngoài lớp hòa
            pred = np.argmax(y_prob[0][:2])  # Chọn lớp thắng (0) hoặc thua (1)
        else:
            # Chọn lớp có xác suất cao nhất
            pred = np.argmax(y_prob)
        
        # Kết quả
        if pred == 0:
            result = f'{team_home} thắng'
        elif pred == 1:
            result = f'{team_away} thắng'
        else:
            result = 'Hòa'

        # Truyền các giá trị vào template
        return render_template('result.html',
                                team_home=team_home,
                                team_away=team_away,
                                result = result,
                                home_win_rate=np.round(features['Tỉ Lệ Thắng Đội Nhà'], 2),
                                away_win_rate=np.round(features['Tỉ Lệ Thắng Đội Khách'],2),
                                home_goals_per_match= (np.ceil(features['Số Bàn Trung Bình Đội Nhà']) if features['Số Bàn Trung Bình Đội Nhà'] - np.floor(features['Số Bàn Trung Bình Đội Nhà']) > 0.5 else np.floor(features['Số Bàn Trung Bình Đội Nhà'])).astype(int),
                                away_goals_per_match= (np.ceil(features['Số Bàn Trung Bình Đội Khách']) if features['Số Bàn Trung Bình Đội Khách'] - np.floor(features['Số Bàn Trung Bình Đội Khách']) > 0.5 else np.floor(features['Số Bàn Trung Bình Đội Khách'])).astype(int),
                                recent_home_form = f"{home_wins} thắng - {home_loses} thua - {home_draws} hòa",
                                recent_away_form = f"{away_wins} thắng - {away_loses} thua - {away_draws} hòa",
                                head_to_head_stats=head_to_head_stats,
                                )
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


