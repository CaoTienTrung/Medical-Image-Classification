## 📌 Tên bài toán  
**Phân loại bệnh ung thư phổi qua ảnh CT**  

## 🌟 Tầm quan trọng của bài toán  
Ung thư phổi là một trong những loại ung thư gây tử vong hàng đầu trên toàn thế giới, với tỷ lệ phát hiện muộn và điều trị không kịp thời còn cao. Việc ứng dụng trí tuệ nhân tạo vào hỗ trợ chẩn đoán hình ảnh y học – đặc biệt là phân tích ảnh CT ngực – có thể giúp bác sĩ phát hiện sớm và phân loại chính xác các dạng ung thư phổi, từ đó đưa ra phác đồ điều trị phù hợp hơn. Hệ thống phân loại tự động không những hỗ trợ nâng cao hiệu quả chẩn đoán, mà còn giúp tiết kiệm thời gian, giảm tải cho đội ngũ y tế, đặc biệt ở những vùng còn thiếu chuyên gia chẩn đoán hình ảnh.

## 🧩 Phát biểu bài toán  

Bài toán được định nghĩa theo cấu trúc **Input – Output** như sau:

- **Input**:  
  • Tập dữ liệu \( X = \{X_1, X_2, X_3,…, X_N\} \) gồm \( N \) ảnh đã được gán nhãn. Mỗi ảnh \( X_i \) có kích thước 224 x 224 pixels, biểu diễn một lát cắt ngực từ ảnh CT của bệnh nhân.  
  • Mỗi ảnh được gắn nhãn thuộc tập nhãn \( L = \{0, 1, 2, 3\} \), trong đó:  
    - 0: Adenocarcinoma (Ung thư biểu mô tuyến)  
    - 1: Large cell carcinoma (Ung thư biểu mô tế bào lớn)  
    - 2: Squamous cell carcinoma (Ung thư biểu mô tế bào vảy)  
    - 3: Normal (Bình thường)  

- **Output**:  
  • Tập nhãn dự đoán \( Y = \{Y_1, Y_2, Y_3,…, Y_N\} \), với mỗi phần tử \( Y_i \in L \) là nhãn đầu ra tương ứng cho ảnh đầu vào \( X_i \).

## 🧠 Mô hình sử dụng

Để giải quyết bài toán, nhóm áp dụng cả các phương pháp học máy cổ điển và học sâu hiện đại, bao gồm:

- **Phương pháp trích xuất đặc trưng cổ điển + ML classifier**  
  - HOG + SVM / XGBoost  
  - SIFT + Random Forest  
  - Gabor Filter + Logistic Regression  
  - Haralick Features + SVM  

- **Phương pháp học sâu sử dụng mô hình tiền huấn luyện**  
  - **VGG16** (pretrained on ImageNet) – sử dụng như feature extractor  
  - **ResNet50** (pretrained on ImageNet) – sử dụng như feature extractor   
  - Các mô hình học sâu được kết hợp với các bộ phân loại như SVM, Logistic Regression, XGBoost để tăng hiệu quả nhận diện và giảm thiểu overfitting do dữ liệu y học hạn chế.

## 🖼️ Minh hoạ kiến trúc mô hình (paste ảnh tại đây)
![Pipeline]([path/to/your/image.png](https://www.google.com/search?q=%E1%BA%A3nh+y+t%E1%BA%BF+x%E1%BB%AD+l%C3%BD+%E1%BA%A3nh+y+khoa&sca_esv=6d6d50e00e08b0cb&rlz=1C1GCEA_enVN1021VN1021&udm=2&biw=1396&bih=663&ei=P0NZaKTVBa_c2roPlI7lsAQ&ved=0ahUKEwjk8OXSwIeOAxUvrlYBHRRHGUYQ4dUDCBE&uact=5&oq=%E1%BA%A3nh+y+t%E1%BA%BF+x%E1%BB%AD+l%C3%BD+%E1%BA%A3nh+y+khoa&gs_lp=EgNpbWciIuG6o25oIHkgdOG6vyB44butIGzDvSDhuqNuaCB5IGtob2FIhRtQiAVYrxlwB3gAkAECmAHKAaABuhaqAQYzLjE5LjG4AQPIAQD4AQGYAgygAqAHwgIFEAAYgATCAgQQABgewgIGEAAYBRgewgIHEAAYgAQYE8ICBhAAGBMYHsICCBAAGBMYBRgewgIGEAAYCBgemAMAiAYBkgcDNS43oAe4F7IHAzEuN7gHmwfCBwM0LjjIBxA&sclient=img#vhid=gBQD-vWFCy98RM&vssid=mosaic))
