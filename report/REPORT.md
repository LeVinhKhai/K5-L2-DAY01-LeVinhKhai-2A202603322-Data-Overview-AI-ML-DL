# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:6/11/2026**

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`=0.510915; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** vì đây là trường task có giá trị là "image classification", model nhận toàn bộ ảnh làm đầu vào và trả về các lớp được xếp hạng theo `score`.
3. Danh sách tên lớp không phải do model tự nghĩ ra, mà được xác định sẵn bởi taxonomy/dataset dùng để huấn luyện checkpoint. Trong record này, `taxonomy_name` là `ImageNet-1K`, nên model chỉ dự đoán trong tập các lớp đã được định nghĩa sẵn của ImageNet-1K.
4. Cần ghi cả `class_id`, `class_name` và `taxonomy_name` để xác định nhãn một cách đầy đủ và không bị mơ hồ. `class_id` cho biết mã lớp trong taxonomy, `class_name` cho biết tên dễ đọc của lớp, còn `taxonomy_name` cho biết lớp đó thuộc bộ nhãn nào. Nếu chỉ ghi mỗi tên lớp thì có thể khó đối chiếu chính xác giữa các dataset hoặc checkpoint khác nhau.
5. Nếu ảnh có nhiều vật hoặc nhiều chủ đề, guideline cần quy định rõ **tiêu chí chọn lớp chính** — ví dụ ưu tiên vật thể chiếm diện tích lớn nhất, vật thể ở trung tâm, vật thể liên quan trực tiếp đến mục tiêu bài toán, hoặc chủ đề nổi bật nhất của ảnh. Nếu không có tiêu chí rõ ràng, các annotator khác nhau có thể chọn các lớp khác nhau cho cùng một ảnh.
6. Không được lấy `score` của model làm ground truth vì `score` chỉ thể hiện mức độ tự tin của model vào prediction, không chứng minh prediction đó đúng. Ground truth là thước đo chuẩn phải đến từ nhãn đã được con người gán theo guideline và được kiểm tra chất lượng độc lập với model.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở tại khoảng vị trí (385.33,69.24) pixel, góc dưới-phải ở vị trí (498.92,348.92) pixel, rộng 113.58 pixel ( lấy x_max - x_min= 498.92-385.33), cao 279.68 pixel (y_max - y_min=348.92 - 69.24  ) 
3. Ngưỡng 0.20 có 17 predictions, và nguỡng 0.60 có 6 predictions.
4. Khi hạ ngưỡng, số lượng prediction tăng nên reviewer phải kiểm tra nhiều bounding box hơn, loại các box sai hoặc không phù hợp, chỉnh lại box chưa chính xác và xác nhận các vật thể thật.
5. Bounding box phải bao sát toàn bộ phần nhìn thấy của vật thể, không cắt mất phần của vật thể và hạn chế lấy nền thừa xung quanh. Cùng một loại vật thể phải được vẽ box theo cùng một nguyên tắc trên tất cả ảnh.
6. Trên ảnh `kitchen`, "Person" được model phát hiện với score khoảng `0.61` ở mép trái ảnh bị cắt mép (truncated) vì chỉ một phần tay nằm trong khung hình, phần còn lại nằm ngoài ảnh. Nếu guideline chưa quy định cách xử lý vật thể bị cắt mép, tôi sẽ escalation lên Lab Coach để thống nhất cách gán nhãn, thay vì tự đặt quy tắc, nhằm đảm bảo các annotator xử lý nhất quán.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= "kitchen-001" ; `class_name`="person"; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`= `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`
2. So với bounding box chỉ cho biết vùng hình chữ nhật bao quanh vật thể, polygon mô tả chi tiết đường biên và hình dạng thực tế của vật thể. Nhờ đó, segmentation có thể xác định chính xác những pixel nào thuộc về vật thể và hạn chế phần nền nằm trong vùng nhãn.
3. `instance_id` dùng để định danh riêng từng đối tượng xuất hiện trong ảnh, giúp phân biệt nhiều instance khác nhau ngay cả khi chúng cùng thuộc một class. Ví dụ, hai người đều có `class_name` = person nhưng mỗi người sẽ có `instance_id` khác nhau. instance_id không phải `class_id` vì nó không biểu diễn loại vật thể, và cũng không phải tracking ID vì bài này không theo dõi cùng một vật thể qua nhiều frame.
4. Khi vẽ mask, đường biên polygon phải bám sát phần nhìn thấy thực tế của vật thể, hạn chế lấy nền thừa và không bỏ sót các phần nhìn thấy rõ. Quy tắc này cần được áp dụng nhất quán cho cùng một loại vật thể trên toàn bộ tập dữ liệu.
5. Trong ảnh `kitchen`, các `bowl` ở góc dưới bên trái nằm sát và che nhau một phần nên biên giữa từng instance khó xác định rõ. Nếu guideline đã quy định cách xử lý vật thể chạm/che nhau thì em sẽ làm theo guideline; nếu chưa có quy định rõ, em sẽ escalation lên Lab Coach để thống nhất cách vẽ mask, tránh mỗi annotator tự xác định biên khác nhau và làm ground truth không nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn bộ ảnh, theo taxonomy đã quy định |Ảnh có nhiều vật/chủ đề nên khó xác định lớp chính; prediction có thể có score cao nhưng vẫn không phải ground truth  | Chọn đúng một lớp theo guideline và taxonomy, không dựa máy móc vào prediction của model | Kiểm tra lớp đã chọn có đúng guideline/taxonomy không và có nhất quán với các ảnh tương tự không |
| Phát hiện vật thể | Mỗi vật thể có một `class` và một bounding box | Có vật thể bị cắt mép/che khuất; threshold thấp tạo thêm nhiều prediction có thể là false positive | Xác định các object cần gán nhãn, chọn đúng class và vẽ box bao sát phần vật thể theo guideline | Kiểm tra thiếu/thừa object, class sai, box lệch/quá rộng/quá hẹp và cách xử lý vật bị che/cắt mép |
| Instance segmentation | Mỗi instance có một `class` và một polygon/mask bao theo biên vật thể | Biên vật thể có thể khó xác định khi vật bị che, chạm nhau hoặc nền phức tạp | Vẽ polygon/mask bám theo phần nhìn thấy của từng instance và gán đúng class | Kiểm tra mask có bám đúng biên, có bị thiếu/thừa vùng, chồng nhầm instance hoặc gán sai class không |

## 5. An toàn dữ liệu

1. Trong bài lab này, em chỉ sử dụng dữ liệu đúng với phạm vi được yêu cầu và không đưa thông tin cá nhân, dữ liệu khách hàng hay các thông tin nhạy cảm như password, token hoặc secret lên notebook, hoặc repo công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi bài lab, em sẽ dừng xử lý và báo lại cho Lab Coach/mentor để được hướng dẫn trước khi tiếp tục.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x ] `REPORT.md` (file này đã điền)
- [x ] `day1_lab_outputs/classification_predictions.json`
- [x ] `day1_lab_outputs/detection_predictions.json`
- [x ] `day1_lab_outputs/segmentation_predictions.json`
- [x ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x ] `day1_lab_outputs/visuals/classification_top5.png`
- [x ] `day1_lab_outputs/visuals/detection_predictions.png`
- [x ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x ] Ô validation cuối notebook báo `PASS`
- [x ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x ] Đã nộp **link repository** trên LMS