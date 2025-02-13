from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
import get_data_regulary as gdr
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


#matches_df = gdr.fetch_data()
matches_df = pd.read_csv('Model_Train/data_to_4-11-2024.csv')

matches_df['Phong Độ Gần Đây Đội Nhà'] = matches_df.groupby('Đội Chủ Nhà')['Kết quả'].transform(lambda x: x.rolling(3, min_periods=1).mean())
matches_df['Phong Độ Gần Đây Đội Khách'] = matches_df.groupby('Đội Khách')['Kết quả'].transform(lambda x: x.rolling(3, min_periods=1).mean())
     
so_tran_thang_doi_nha = matches_df[matches_df['Kết quả'] == 0].groupby('Đội Chủ Nhà').size()
tong_so_tran_doi_nha = matches_df.groupby('Đội Chủ Nhà').size()
ti_le_thang_doi_nha = (so_tran_thang_doi_nha / tong_so_tran_doi_nha).fillna(0)
matches_df['Tỉ Lệ Thắng Đội Nhà'] = matches_df['Đội Chủ Nhà'].map(ti_le_thang_doi_nha)

so_tran_thang_doi_khach = matches_df[matches_df['Kết quả'] == 1].groupby('Đội Khách').size()  
tong_so_tran_doi_khach = matches_df.groupby('Đội Khách').size()  
ti_le_thang_doi_khach = (so_tran_thang_doi_khach / tong_so_tran_doi_khach).fillna(0)
matches_df['Tỉ Lệ Thắng Đội Khách'] = matches_df['Đội Khách'].map(ti_le_thang_doi_khach)

so_tran_hoa_doi_nha = matches_df[matches_df['Kết quả'] == 2].groupby('Đội Chủ Nhà').size()  # Số trận hòa của đội chủ nhà
ti_le_hoa_doi_nha = (so_tran_hoa_doi_nha / tong_so_tran_doi_nha).fillna(0)
matches_df['Tỉ Lệ Hòa Đội Nhà'] = matches_df['Đội Chủ Nhà'].map(ti_le_hoa_doi_nha)

so_tran_hoa_doi_khach = matches_df[matches_df['Kết quả'] == 2].groupby('Đội Khách').size()  # Số trận hòa của đội khách
ti_le_hoa_doi_khach = (so_tran_hoa_doi_khach / tong_so_tran_doi_khach).fillna(0)
matches_df['Tỉ Lệ Hòa Đội Khách'] = matches_df['Đội Khách'].map(ti_le_hoa_doi_khach)

matches_df['Số Bàn Trung Bình Đội Nhà'] = matches_df.groupby('Đội Chủ Nhà')['Bàn Thắng Đội Nhà'].transform('mean')
matches_df['Số Bàn Trung Bình Đội Khách'] = matches_df.groupby('Đội Khách')['Bàn Thắng Đội Khách'].transform('mean')
matches_df['Hiệu Số Bàn Thắng Đội Nhà'] = matches_df.groupby('Đội Chủ Nhà')['Bàn Thắng Đội Nhà'].transform('mean') - matches_df.groupby('Đội Chủ Nhà')['Bàn Thắng Đội Khách'].transform('mean')
matches_df['Hiệu Số Bàn Thắng Đội Khách'] = matches_df.groupby('Đội Khách')['Bàn Thắng Đội Khách'].transform('mean') - matches_df.groupby('Đội Khách')['Bàn Thắng Đội Nhà'].transform('mean')

# Thêm các đặc trưng thống kê và đối đầu
matches_df['Sạch Lưới Đội Nhà'] = matches_df.groupby('Đội Chủ Nhà')['Bàn Thắng Đội Khách'].transform(lambda x: (x == 0).sum())
matches_df['Sạch Lưới Đội Khách'] = matches_df.groupby('Đội Khách')['Bàn Thắng Đội Nhà'].transform(lambda x: (x == 0).sum())
matches_df['Thành Tích Đối Đầu Đội Nhà'] = matches_df.groupby(['Đội Chủ Nhà', 'Đội Khách'])['Kết quả'].transform(lambda x: (x == 0).sum())
matches_df['Thành Tích Đối Đầu Đội Khách'] = matches_df.groupby(['Đội Khách', 'Đội Chủ Nhà'])['Kết quả'].transform(lambda x: (x == 1).sum())

# Tạo các đặc trưng mới từ sự chênh lệch bàn thắng, chuỗi thắng/thua gần đây
matches_df['Chênh Lệch Bàn Thắng Nhà-Khách'] = matches_df['Hiệu Số Bàn Thắng Đội Nhà'] - matches_df['Hiệu Số Bàn Thắng Đội Khách']
matches_df['Chênh Lệch Phong Độ Gần Đây'] = matches_df['Phong Độ Gần Đây Đội Nhà'] - matches_df['Phong Độ Gần Đây Đội Khách']
matches_df['Chuỗi Thắng Đội Nhà'] = matches_df.groupby('Đội Chủ Nhà')['Kết quả'].transform(lambda x: x.eq(0).rolling(5, min_periods=1).sum())
matches_df['Chuỗi Thắng Đội Khách'] = matches_df.groupby('Đội Khách')['Kết quả'].transform(lambda x: x.eq(1).rolling(5, min_periods=1).sum())

# Mã hóa tên đội bóng
le = LabelEncoder()
matches_df['Đội Chủ Nhà'] = le.fit_transform(matches_df['Đội Chủ Nhà'])
matches_df['Đội Khách'] = le.transform(matches_df['Đội Khách'])

X = matches_df[['Đội Chủ Nhà', 'Đội Khách', 'Phong Độ Gần Đây Đội Nhà', 'Phong Độ Gần Đây Đội Khách', 
                'Tỉ Lệ Thắng Đội Nhà', 'Tỉ Lệ Thắng Đội Khách', 'Tỉ Lệ Hòa Đội Nhà',
               'Tỉ Lệ Hòa Đội Khách', 'Số Bàn Trung Bình Đội Nhà', 'Số Bàn Trung Bình Đội Khách',
               'Hiệu Số Bàn Thắng Đội Nhà', 'Hiệu Số Bàn Thắng Đội Khách', 'Sạch Lưới Đội Nhà',
               'Sạch Lưới Đội Khách', 'Thành Tích Đối Đầu Đội Nhà', 'Thành Tích Đối Đầu Đội Khách',
               'Chênh Lệch Bàn Thắng Nhà-Khách', 'Chênh Lệch Phong Độ Gần Đây',
               'Chuỗi Thắng Đội Nhà', 'Chuỗi Thắng Đội Khách']]

# Nhân trọng số cho Tỉ Lệ Thắng để tăng cường ảnh hưởng
X.loc[:, 'Tỉ Lệ Thắng Đội Nhà'] *= 10
X.loc[:, 'Tỉ Lệ Thắng Đội Khách'] *= 10

y = matches_df['Kết quả']

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Áp dụng SMOTE để cân bằng lớp
smote = SMOTE(sampling_strategy={0: int(len(y) * 0.5), 
                                1: int(len(y) * 0.5), 
                                2: int(len(y) * 0.3)}, random_state=42)

X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Áp dụng trọng số lớp để giảm độ thiên vị về dự đoán hòa
class_weights = {0: 2, 1: 1.5, 2: 1}

# Mô hình RandomForest với trọng số lớp
rf_model = RandomForestClassifier(random_state=42, class_weight=class_weights, 
                                  bootstrap=False, max_depth=15, min_samples_leaf=2, 
                                  min_samples_split=2, n_estimators=500)

# Mô hình XGBoost
xgb_model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', 
                          random_state=42)

# Sử dụng Stacking để kết hợp mô hình RandomForest và XGBoost
estimator = [('rf', rf_model), ('xgb', xgb_model)]
stacking_model = StackingClassifier(estimators=estimator, final_estimator=LogisticRegression())

# Huấn luyện mô hình
stacking_model.fit(X_balanced, y_balanced)


# Lưu mô hình
joblib.dump(stacking_model, "stacking_model.joblib")
