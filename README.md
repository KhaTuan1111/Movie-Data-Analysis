# 🎬 Khám Phá Dữ Liệu & Phân Tích Chuyên Sâu: 9.500+ Phim Phổ Biến Trên TMDB

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-TMDB%209500+-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/tushargoel04/9500plus-popular-movies-tmdb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎓 Đồ Án Cuối Kỳ Môn Học (Chiếm 40% Tổng Điểm)
- **Môn học**: Khai Phá & Phân Tích Dữ Liệu (Exploratory Data Analysis & Data Science)
- **Hình thức**: Đồ án nhóm (2–3 sinh viên)
- **Quản lý phiên bản & Làm việc nhóm**: Git & GitHub

### 👥 Danh Sách Thành Viên Nhóm
| Mã Số Sinh Viên | Họ và Tên | Vai Trò & Phân Công Nhiệm Vụ |
| :---: | :---: | :--- |
| `24120xxx` | **Thành viên 1 (Nhóm trưởng)** | Thu thập dữ liệu, Đánh giá chất lượng, Tiền xử lý quan hệ & Phân tích Câu 1 – Câu 2 |
| `24120xxx` | **Thành viên 2** | Mô hình hóa thống kê, Trích xuất đặc trưng (Feature Engineering) & Phân tích Câu 3 – Câu 4 |
| `24120xxx` | **Thành viên 3** | Phân tích Studio / Ngôn ngữ, Xây dựng Web Dashboard Streamlit & Phân tích Câu 5 – Câu 6 |

---

## 📌 Giới Thiệu & Đặt Vấn Đề
Ngành công nghiệp điện ảnh toàn cầu là một thị trường giải trí trị giá hàng trăm tỷ USD với rủi ro tài chính rất lớn. Các quyết định đầu tư dao động từ vài trăm nghìn USD cho phim độc lập đến hơn 350 triệu USD cho các tác phẩm bom tấn thương hiệu. Việc thấu hiểu các yếu tố chi phối lợi nhuận phòng vé, tỷ suất hoàn vốn (ROI), thị hiếu khán giả và thời điểm công chiếu là điều tối quan trọng đối với các đạo diễn, nhà sản xuất và nhà đầu tư.

Đồ án này nghiên cứu bộ dữ liệu **[9500+ Popular Movies TMDB](https://www.kaggle.com/datasets/tushargoel04/9500plus-popular-movies-tmdb)** thu thập từ Kaggle. Thông qua quy trình khoa học dữ liệu hoàn chỉnh, nhóm phát hiện các lỗi cấu trúc dữ liệu tiềm ẩn, làm sạch và chuẩn hóa dữ liệu, giải quyết triệt để 6 câu hỏi nghiên cứu then chốt trong ngành công nghiệp điện ảnh.

---

## 🔍 Phát Hiện Bất Thường Dữ Liệu & Chuẩn Hóa Cấu Trúc (Data Profiling & Quirks)
Trong giai đoạn khám phá ban đầu, nhóm đã phát hiện 3 vấn đề cấu trúc cốt lõi trong file thô `9616 UNIQUE IMDB.csv`:

1. **Lỗi tích chập dòng Cartesian (Exploded Cross-Product Quirk)**:
   - File dữ liệu thô có **83.739 dòng**, nhưng thực chất chỉ có **9.961 bộ phim duy nhất**.
   - **Nguyên nhân**: Dữ liệu bị tách dòng (unnest) chéo giữa thể loại (`genres`) và hãng sản xuất (`production_companies`). Ví dụ: Phim *The Pope's Exorcist* có 3 thể loại và 6 hãng sản xuất $\rightarrow$ bị nhân bản thành $3 \times 6 = 18$ dòng giống hệt nhau về ngân sách và doanh thu.
   - *Nếu tính tổng (`sum`) hoặc trung bình (`mean`) trực tiếp trên file gốc, doanh thu và ngân sách sẽ bị thổi phồng sai lệch từ 5 lần đến 30 lần.*
2. **Giá trị ngân sách và doanh thu bằng 0 (`budget = 0`, `revenue = 0`)**:
   - Khoảng 48% phim có ngân sách = 0 và 45% có doanh thu = 0. Đây là dữ liệu chưa được công bố/thu thập (missing values) chứ không phải phim miễn phí. Nhóm đã tạo cờ `has_financial_data` để lọc chính xác khi phân tích tài chính.
3. **Mã ngôn ngữ không đồng nhất**:
   - Tồn tại đồng thời mã viết tắt `'cn'` và tên tiếng Anh `'Chinese'`, được nhóm quy chuẩn về một tên chuẩn duy nhất.

### Kiến Trúc Dữ Liệu Quan Hệ (Normalized Schema)
Để khắc phục lỗi trùng lặp, nhóm đã chuẩn hóa dữ liệu thành 3 bảng:
- **`movies_cleaned.csv`** (9.961 dòng): Mỗi dòng đại diện cho 1 bộ phim duy nhất, chứa thông tin siêu dữ liệu, danh sách gộp thể loại/hãng sản xuất, thời gian đã bóc tách, lợi nhuận và ROI.
- **`movies_genres.csv`** (25.712 dòng): Bảng cầu nối quan hệ 1-N giữa `movie_id` $\leftrightarrow$ `genre`.
- **`movies_companies.csv`** (30.931 dòng): Bảng cầu nối quan hệ 1-N giữa `movie_id` $\leftrightarrow$ `production_company`.

---

## ❓ 6 Câu Hỏi Nghiên Cứu & Kết Quả Phân Tích Cốt Lõi

### 🎯 Câu hỏi 1: Thể loại nào đem lại hiệu quả tài chính và tỷ suất sinh lời (ROI) cao nhất?
- **Dẫn đầu về doanh thu tuyệt đối**: **Hoạt hình (Animation)** (trung bình >260 triệu USD), **Phiêu lưu (Adventure)** (>245 triệu USD) và **Khoa học viễn tưởng (Sci-Fi)**.
- **Dẫn đầu về hiệu quả vốn (ROI)**: **Kinh dị (Horror)** và **Bí ẩn (Mystery)** đạt **median ROI vượt 200%–250%**. Phim kinh dị có kinh phí sản xuất rất thấp (trung vị 10–15 triệu USD) nhưng tỷ lệ nhân vốn cực lớn, là kênh đầu tư an toàn và sinh lời nhất.

### ⏳ Câu hỏi 2: Điện ảnh đã tiến hóa ra sao qua các thập kỷ (1920–2023)?
- **Bùng nổ số lượng**: Hơn 70% số phim được sản xuất sau năm 2000 nhờ sự phổ biến của máy quay kỹ thuật số và các nền tảng phân phối trực tuyến.
- **Ngân sách leo thang**: Ngân sách trung vị tăng từ <5 triệu USD (thập niên 1960) lên hơn 40 triệu USD (giai đoạn 2010–2023).
- **"Tỷ lệ vàng" thời lượng**: Xuyên suốt 9 thập kỷ, thời lượng phim chiếu rạp trung vị luôn giữ ổn định kỳ lạ ở mức **98 – 108 phút**, phù hợp với chu kỳ chú ý của con người và lịch sắp xếp suất chiếu của rạp.

### 📅 Câu hỏi 3: Thời điểm phát hành (Tháng / Mùa) tác động thế nào đến doanh thu?
- **Hai mùa vàng phòng vé**:
  - **Mùa phim hè (Tháng 5 – 7)**: Đạt đỉnh doanh thu (trung bình 150–175 triệu USD/phim) nhờ kỳ nghỉ hè của học sinh/sinh viên và các phim bom tấn.
  - **Mùa lễ hội cuối năm (Tháng 11 – 12)**: Tăng vọt nhờ dịp Lễ Tạ Ơn, Giáng Sinh và chiến dịch vận động giải thưởng điện ảnh.
- **Tháng trũng phòng vé ("Dump Months")**: Tháng 1 và tháng 9 có doanh thu thấp nhất (~70–85 triệu USD), thường chỉ dành cho phim kinh phí nhỏ hoặc phim thể nghiệm.

### ⭐ Câu hỏi 4: Điểm đánh giá của khán giả vs. Sức hút thương mại?
- **Tiền mua được sự phổ biến, không mua được sự yêu mến**:
  - Ngân sách có tương quan rất mạnh với Doanh thu ($r \approx 0.72$), chứng minh quy mô kinh phí và truyền thông quyết định lượng vé bán ra.
  - Tuy nhiên, tương quan giữa Ngân sách và Điểm đánh giá (`vote_average`) gần như bằng 0 ($r \approx 0.05$). Đổ nhiều tiền vào kỹ xảo không đồng nghĩa với việc phim sẽ được khán giả yêu thích. Nhiều tác phẩm điểm cao nhất lại có kinh phí rất vừa phải.

### 🏢 Câu hỏi 5: Hãng phim nào thống trị doanh thu và hãng nào đạt tỷ suất lợi nhuận cao nhất?
- **Thống trị tổng doanh thu lịch sử**: Các hãng phim lâu đời như **Warner Bros., Universal Pictures, Walt Disney Pictures và Columbia Pictures** (tổng doanh thu mỗi hãng vượt 50–80 tỷ USD nhờ kho phim đồ sộ).
- **Thống trị lợi nhuận trên mỗi phim**: **Marvel Studios** và **Pixar** đạt mức lợi nhuận trung bình trên mỗi phim vượt **350–500 triệu USD**, vượt trội hoàn toàn so với chuẩn chung của ngành.

### 🌍 Câu hỏi 6: Điện ảnh quốc tế ngoài tiếng Anh thể hiện ra sao?
- **Chất lượng vượt trội của phim quốc tế**: Phim tiếng **Nhật** (điểm TB ~7.30) và tiếng **Hàn** (~7.15) có điểm đánh giá trung bình cao hơn hẳn so với phim tiếng Anh (~6.48). Điều này phản ánh các tác phẩm châu Á gây tiếng vang toàn cầu trên TMDB đều có chiều sâu nghệ thuật, kịch bản độc đáo hoặc cộng đồng hâm mộ trung thành (như Anime Nhật Bản, Thriller Hàn Quốc).

---

## 📊 Danh Mục Biểu Đồ Trực Quan Hóa (Báo Cáo)

| Câu hỏi phân tích | Tên file biểu đồ xuất bản (300 DPI) | Nội dung trực quan |
| :--- | :--- | :--- |
| **Q1: Thể loại & ROI** | `reports/figures/q1_genre_finances.png` | So sánh Doanh thu trung bình & Tỷ suất ROI trung vị theo Thể loại |
| **Q2: Tiến hóa theo thập kỷ** | `reports/figures/q2_decade_trends.png` | Xu hướng số lượng phim, tăng trưởng ngân sách & thời lượng qua các thập kỷ |
| **Q3: Tính mùa vụ** | `reports/figures/q3_seasonality.png` | Doanh thu trung bình theo tháng & Lợi nhuận ròng theo mùa phát hành |
| **Q4: Tương quan Ratings & Doanh thu** | `reports/figures/q4_correlations.png` | Ma trận tương quan nhiệt (Heatmap) & Phân tán Ngân sách vs Doanh thu |
| **Q5: Bảng xếp hạng Studio** | `reports/figures/q5_top_studios.png` | Top 15 Studio theo Tổng doanh thu phòng vé & Lợi nhuận trung bình |
| **Q6: Bản đồ điện ảnh ngôn ngữ** | `reports/figures/q6_language_landscape.png` | Điểm đánh giá trung bình & Biểu đồ hộp (Boxplot) phân phối điểm theo ngôn ngữ |

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
Movie-Data-Analysis/
├── data/
│   ├── raw/
│   │   └── 9616_UNIQUE_IMDB.csv          # Dữ liệu gốc từ Kaggle (83.739 dòng)
│   └── processed/
│       ├── movies_cleaned.csv            # 9.961 phim đã làm sạch và khử trùng lặp
│       ├── movies_genres.csv             # Bảng liên kết Phim - Thể loại (25.712 dòng)
│       └── movies_companies.csv          # Bảng liên kết Phim - Hãng phim (30.931 dòng)
├── notebooks/
│   └── movie_data_analysis.ipynb         # Jupyter Notebook tiếng Việt đã chạy đầy đủ kết quả
├── reports/
│   └── figures/                          # Toàn bộ biểu đồ độ phân giải cao phục vụ báo cáo
│       ├── q1_genre_finances.png
│       ├── q2_decade_trends.png
│       ├── q3_seasonality.png
│       ├── q4_correlations.png
│       ├── q5_top_studios.png
│       └── q6_language_landscape.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Module tải dữ liệu tự động
│   ├── preprocessor.py                   # Module tiền xử lý và khử trùng lặp
│   └── analysis_utils.py                 # Hàm phân tích và vẽ biểu đồ tự động
├── scripts/
│   └── build_notebook.py                 # Script tự động tạo Jupyter Notebook tiếng Việt
├── app.py                                # Web Dashboard tương tác (Streamlit) để thuyết trình
├── requirements.txt                      # Danh sách các thư viện cần thiết
├── .gitignore                            # Cấu hình loại trừ file rác Git
└── README.md                             # Báo cáo tổng thể dự án bằng tiếng Việt
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Dự Án

### 1. Cài đặt môi trường
Mở terminal và clone dự án về máy:
```bash
git clone https://github.com/KhaTuan1111/Movie-Data-Analysis.git
cd Movie-Data-Analysis
python -m pip install -r requirements.txt
```

### 2. Thực hiện tiền xử lý dữ liệu (Nếu muốn chạy lại từ đầu)
Chạy script tiền xử lý để làm sạch dữ liệu thô:
```bash
python src/preprocessor.py
```

### 3. Tự động xuất toàn bộ biểu đồ báo cáo
Tạo lại 6 biểu đồ trong thư mục `reports/figures/`:
```bash
python src/analysis_utils.py
```

### 4. Mở Jupyter Notebook để xem phân tích chi tiết
Khởi động Jupyter Notebook tiếng Việt:
```bash
jupyter notebook notebooks/movie_data_analysis.ipynb
```

### 5. Khởi động Web App Dashboard để thuyết trình trước Giảng viên
Chạy giao diện tương tác:
```bash
streamlit run app.py
```

---

## 💡 Đề Xuất Chiến Lược Cho Nhà Sản Xuất & Nhà Đầu Tư
1. **Chiến lược đầu tư quả tạ (The Barbell Strategy)**:
   - Dành tỷ trọng đầu tư vào thể loại **Kinh dị & Bí ẩn** với ngân sách thấp (5–15 triệu USD) để đảm bảo lợi nhuận vốn an toàn (median ROI > 200%).
   - Chỉ rót ngân sách lớn (>100 triệu USD) cho các thương hiệu có sẵn (franchise IP) thuộc thể loại **Hoạt hình hoặc Phiêu lưu** công chiếu vào mùa hè.
2. **Kỷ luật lịch phát hành**:
   - Tránh công chiếu các phim trọng điểm vào tháng 1 và tháng 9.
   - Ưu tiên hai cửa sổ vàng: **Mùa hè (Tháng 5–7)** hoặc **Dịp lễ cuối năm (Tháng 11–12)**.
3. **Đầu tư vào kịch bản hơn là chỉ chạy đua kỹ xảo**:
   - Kinh phí lớn không đồng nghĩa với điểm đánh giá cao của khán giả. Chất lượng kịch bản, diễn xuất và đạo diễn mới là yếu tố quyết định giá trị nghệ thuật lâu dài.

---
*Dự án phục vụ đánh giá học phần Đồ án môn học (Group Final Project).*