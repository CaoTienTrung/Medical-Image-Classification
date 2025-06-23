## **Phân loại bệnh ung thư phổi qua ảnh CT**   
![Pipeline minh họa](https://daotaolientuc.edu.vn/wp-content/uploads/2021/07/nganh-ky-thuat-hinh-anh-y-hoc-3.jpg)
## 🌟 Tầm quan trọng của bài toán  
Trong thời đại công nghiệp phát triển dẫn đến không khí ngày càng ô nhiễm, sức khỏe của hệ hô hấp của mỗi người ngày càng bị đe dọa. Các bệnh về phổi, đặc biệt là ung thư phổi đang trở thành 1 chủ đề đáng quan ngại kể cả với những người sống lành mạnh nhất.
Theo American Cancer Society [1], dự kiến trong năm 2025 ở Mỹ sẽ có:
- Khoảng 226,650 ca ung thư phổi mới (110.680 đàn ông, 115.970 phụ nữ)
- Khoảng 124,730 ca tử vong vì ung thư phổi (64.190 đàn ông, 60.540 phụ nữ)
Con số này ước tính chiếm khoảng 20% tổng số ca tử vong vì ung thư, dẫn đầu trong  các loại bệnh ung thư ở Mỹ. Hầu hết các ca bệnh được phát hiện ở người trung niên và cao tuổi, đối tượng có sức khỏe yếu hơn so với các thanh niên, điều này càng gây nguy hiểm hơn cho các bệnh nhân bị bệnh ung thư phổi.
Cũng theo American Cancer Society [2], dựa vào cở sở dữ liệu Surveillance, Epidemiology, and End Results (SEER), được duy trì bởi National Cancer Institute (NCI), có thể thống kê được từ năm 2012 – 2018:
- Đối với ung thư phổi không tế bào nhỏ (non-small cell lung cancer), tỉ lệ sống sót trung bình trong vòng 5 năm sau khi được chuẩn đoán chỉ bằng khoảng 28% với 1 người có điều kiện bình thường.
- Đối với ung thư phổi tế bào nhỏ (small cell lung cancer), con số này còn đáng quan ngại hơn, chỉ 7%, tức tỉ lệ người đó sống sót được 5 năm tới chỉ bằng 7% so với khi người đó không bị bệnh.
Từ các hậu quả nghiêm trọng của bệnh ung thư phổi kể trên, việc chẩn đoán sớm và chính xác các bệnh lý này có vai trò quan trọng trong việc nâng cao hiệu quả điều trị cũng như bảo vệ sức khỏe, tính mạng của bệnh nhân. 
Tuy nhiên, ung thư phổi là 1 căn bệnh rất khó để chuẩn đoán. Giai đoạn đầu của bệnh có các biến chứng dễ bị nhầm lẫn với các bệnh khác như tức ngực, ho, mệt mỏi,…. Việc chuẩn đoán ung thư phổi cũng thường phải được thực hiện qua nhiều phương pháp để cho kết quả chính xác như chụp X-Quang, chụp CT, nội soi phế quản, xét nghiệm đờm,… Ngoài ra còn cần có bác sĩ chuyên môn cao để đưa ra được quyết định chuẩn đoán. Điều này làm chi phí của việc khám bệnh trở nên rất tốn kém về cả thời gian lẫn tiền bạc.
Từ đó, việc có 1 mô hình xác định và phân loại các loại bệnh ung thư phổi đáng tin cậy thông qua ảnh chụp CT là rất cần thiết để nâng cao khả năng phòng bệnh, phát hiện bệnh sớm cũng như giảm rất nhiều chi phí cho khám bệnh. 

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

- **Phương pháp trích xuất đặc trưng cổ điển**  
  - Histogram of Oriented Gradients (HOG)
  - Local Binary Pattern (LBP) 
  - Gabor Filter (GF)
  - Scale-Invariant Feature Transform (SIFT)

- **Phương pháp học sâu sử dụng mô hình tiền huấn luyện**  
  - VGG16 (pretrained on ImageNet) – sử dụng như feature extractor  
  - ResNet50 (pretrained on ImageNet) – sử dụng như feature extractor
  - **MIAFEX**
    
- **Các mô hình Machine Learning** 
  - Sử dụng các mô hình học máy cơ bản: Logistic Regression, Random Forest, Support Vector Machine để tăng hiệu quả nhận diện và giảm thiểu overfitting do dữ liệu y học hạn chế.
