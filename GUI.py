import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import math

# Đọc dữ liệu
data = pd.read_csv('Gold_Price.csv') 
dt_train, dt_test = train_test_split(data, test_size=0.3, shuffle=True)

# Chia dữ liệu
X_train = dt_train.drop(['Date', 'Price'], axis=1) 
y_train = dt_train['Price']
X_test = dt_test.drop(['Date', 'Price'], axis=1)
y_test = dt_test['Price']

# Mô hình Linear Regression
reg = LinearRegression().fit(X_train, y_train)

# Mô hình Lasso Regression
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

# Mô hình Neural Network
nn = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000).fit(X_train, y_train)

# Các hàm tính toán
def NSE(y_test, y_predict):  # cang gan 1 cang tot
    return (1 - (np.sum((y_predict - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def MAE(y_test, y_predict):  # cang nho cang tot
    return mean_absolute_error(y_test, y_predict)

# Giao diện người dùng với Streamlit
st.title("Gold Price Prediction")

st.write("Nhập thông tin vào các ô bên dưới để dự đoán giá vàng:")

# Nhập dữ liệu từ người dùng
open_value = st.text_input("Giá tại thời điểm mở cửa thị trường:", "")
high_value = st.text_input("Giá cao nhất trong ngày:", "")
low_value = st.text_input("Giá thấp nhất trong ngày:", "")
volume_value = st.text_input("Khối lượng giao dịch:", "")
chg_value = st.text_input("% Thay đổi so với giá trước đó:", "")

# Nút dự đoán
if st.button("Dự đoán"):
    if open_value and high_value and low_value and volume_value and chg_value:
        try:
            # Chuyển đổi giá trị đầu vào thành mảng numpy
            X_input = np.array([float(open_value), float(high_value), float(low_value), float(volume_value), float(chg_value)]).reshape(1, -1)
            
            # Dự đoán với Linear Regression
            y_input_predict_lr = reg.predict(X_input)
            st.success(f"Kết quả dự đoán theo LinearRegression: {y_input_predict_lr[0]:.2f}")
            
            # Dự đoán với Lasso Regression
            y_input_predict_lasso = lasso.predict(X_input)
            st.success(f"Kết quả dự đoán theo Lasso: {y_input_predict_lasso[0]:.2f}")
            
            # Dự đoán với Neural Network
            y_input_predict_nn = nn.predict(X_input)
            st.success(f"Kết quả dự đoán theo Neural Network: {y_input_predict_nn[0]:.2f}")
            
        except ValueError:
            st.error("Vui lòng nhập đúng định dạng số!")
    else:
        st.warning("Hãy nhập đầy đủ thông tin!")

# Dự đoán trên tập test và hiển thị kết quả cho từng mô hình
st.write("Tỉ lệ dự đoán đúng trên tập: ")

# Linear Regression
y_predict_lr = reg.predict(X_test)
st.write("**Linear Regression**")
st.write(f"R2: {r2_score(y_test, y_predict_lr):.2f}")
st.write(f"NSE: {NSE(y_test, y_predict_lr):.2f}")
st.write(f"MAE: {MAE(y_test, y_predict_lr):.2f}")
st.write(f"RMSE: {math.sqrt(mean_squared_error(y_test, y_predict_lr)):.2f}")

# Lasso
y_predict_lasso = lasso.predict(X_test)
st.write("**Lasso Regression**")
st.write(f"R2: {r2_score(y_test, y_predict_lasso):.2f}")
st.write(f"NSE: {NSE(y_test, y_predict_lasso):.2f}")
st.write(f"MAE: {MAE(y_test, y_predict_lasso):.2f}")
st.write(f"RMSE: {math.sqrt(mean_squared_error(y_test, y_predict_lasso)):.2f}")

# Neural Network
y_predict_nn = nn.predict(X_test)
st.write("**Neural Network**")
st.write(f"R2: {r2_score(y_test, y_predict_nn):.2f}")
st.write(f"NSE: {NSE(y_test, y_predict_nn):.2f}")
st.write(f"MAE: {MAE(y_test, y_predict_nn):.2f}")
st.write(f"RMSE: {math.sqrt(mean_squared_error(y_test, y_predict_nn)):.2f}")
