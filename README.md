# Customer Analytics Project: RFM Segmentation, Revenue Prediction & Churn Classification

Dự án phân tích dữ liệu khách hàng theo pipeline 4 giai đoạn: làm sạch dữ liệu, phân cụm khách hàng (RFM + KMeans), dự đoán doanh thu và dự đoán khách hàng rời bỏ (churn).

## Mục lục
- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Chi tiết từng phase](#chi-tiết-từng-phase)
- [Cách chạy](#cách-chạy)
- [Kết quả chính](#kết-quả-chính)
- [Hạn chế & hướng phát triển](#hạn-chế--hướng-phát-triển)

## Tổng quan

Pipeline gồm 4 notebook chạy tuần tự, mỗi notebook là một phase độc lập:

| Phase | Notebook | Mục tiêu | Output |
|---|---|---|---|
| 0 | `Phase_0_Data_Cleaning.ipynb` | Làm sạch dữ liệu giao dịch, lọc đơn hàng thành công | `data/df_success.csv` |
| 1 | `Phase_1_Clustering.ipynb` | Tính RFM, phân cụm khách hàng bằng KMeans | `data/customer_segmented.csv`, `models/kmeans_model.pkl` |
| 2 | `Phase_2_Money_Regressor.ipynb` | Dự đoán doanh thu 6 tháng cuối từ dữ liệu 6 tháng đầu | `models/regressor_model.pkl` |
| 3 | `Phase_3_Churn_Classification.ipynb` | Dự đoán khách hàng có nguy cơ rời bỏ (churn) | `models/churn_model.pkl` |

## Cấu trúc dự án
project/
├── data/
│ ├── customer_lifetime_value.csv # dữ liệu gốc
│ ├── df_success.csv # sau khi làm sạch (Phase 0)
│ ├── customer.csv # bảng khách hàng
│ └── customer_segmented.csv # sau khi phân cụm (Phase 1)
├── models/
│ ├── kmeans_model.pkl
│ ├── scaler.pkl
│ ├── regressor_model.pkl
│ └── churn_model.pkl
├── notebooks/
│ ├── Phase_0_Data_Cleaning.ipynb
│ ├── Phase_1_Clustering.ipynb
│ ├── Phase_2_Money_Regressor.ipynb
│ └── Phase_3_Churn_Classification.ipynb
└── README.md

## Chi tiết từng phase

### Phase 0 — Data Cleaning
Đọc dữ liệu giao dịch gốc (`customer_lifetime_value.csv`), lọc ra các đơn hàng được đặt thành công, xuất ra `df_success.csv` làm input cho các phase sau.

### Phase 1 — Customer Segmentation (RFM + KMeans)
- Tính chỉ số **RFM** (Recency – Frequency – Monetary) cho từng khách hàng.
- Tách riêng nhóm **Siêu VIP** trước khi đưa vào phân cụm (tránh outlier làm lệch cluster).
- Chuẩn hóa dữ liệu, tìm số cụm tối ưu bằng **Elbow method**.
- Phân cụm bằng **KMeans**, phân tích đặc điểm từng cụm và gán nhãn phân khúc (segment).
- Trực quan hóa cụm bằng **PCA**.
- Lưu kết quả phân khúc và model (KMeans, scaler).

### Phase 2 — Revenue Regression
- Dùng dữ liệu giao dịch **6 tháng đầu** để tạo feature, dự đoán doanh thu **6 tháng cuối** (chia theo mốc thời gian, không chia ngẫu nhiên — tránh rò rỉ dữ liệu tương lai).
- Chuẩn hóa feature, so sánh nhiều model: **Linear Regression, Random Forest, Gradient Boosting**.
- Chọn model tốt nhất, phân tích feature importance, đánh giá trên test set.
- Lưu model kèm scaler, tên feature và log1p transformer cho target.

### Phase 3 — Churn Classification
- Tạo feature RFM và nhãn churn dựa trên hành vi mua hàng.
- EDA khám phá dữ liệu, chia train/test.
- Huấn luyện và so sánh 2 model: **Logistic Regression** và **Random Forest**.
- Đánh giá bằng confusion matrix, phân tích feature importance.
- **Kết quả:** Logistic Regression cho kết quả tốt hơn. Tỷ lệ churn thực tế trong dữ liệu là **29.4%**. Model dự đoán churn dựa chủ yếu vào Recency và Frequency — hợp lý vì khách lâu không mua và mua ít lần có xu hướng rời bỏ cao hơn.

## Cách chạy

```bash
# Cài đặt thư viện cần thiết
pip install pandas numpy matplotlib seaborn scikit-learn joblib

# Chạy tuần tự theo thứ tự phase
jupyter notebook notebooks/Phase_0_Data_Cleaning.ipynb
jupyter notebook notebooks/Phase_1_Clustering.ipynb
jupyter notebook notebooks/Phase_2_Money_Regressor.ipynb
jupyter notebook notebooks/Phase_3_Churn_Classification.ipynb
```

**Lưu ý:** các notebook đọc dữ liệu qua đường dẫn tương đối `../data/...` và lưu model vào `../models/...`, nên cần giữ đúng cấu trúc thư mục ở trên.

## Kết quả chính

- Phân khúc khách hàng rõ ràng theo RFM, tách riêng được nhóm khách hàng giá trị cao (Siêu VIP/Champions).
- Dự đoán được doanh thu 6 tháng tới của khách hàng dựa trên hành vi 6 tháng trước.
- Xác định được nhóm khách hàng có nguy cơ churn cao (~29.4% tỷ lệ churn thực tế) để hỗ trợ chiến lược giữ chân khách hàng.

## Hạn chế & hướng phát triển

- Mới thử nghiệm 2 model cho bài toán churn (Logistic Regression, Random Forest), threshold mặc định 0.5, chưa tối ưu hyperparameter.
- Chưa áp dụng GridSearch/RandomizedSearch để tune model.
- Chưa xử lý mất cân bằng dữ liệu (có thể thử SMOTE cho bài toán churn).
- Có thể mở rộng thêm các model khác (XGBoost, LightGBM) để so sánh hiệu năng.
