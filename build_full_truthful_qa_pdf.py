import os

tex_content = r"""\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[vietnamese,english]{babel}
\usepackage[T5]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1.8cm,top=2.2cm,bottom=2.0cm,headheight=24pt]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage{tcolorbox}
\usepackage{enumitem}

\emergencystretch=2.0em

\hypersetup{
    colorlinks=true,
    linkcolor=blue!80!black,
    citecolor=blue!70!black,
    urlcolor=blue!80!black,
    pdftitle={So Tay Thuyet Trinh 8 Phut va Phan Bien Bao Ve Deep Learning (CO3039)},
    pdfauthor={Nhom Nghien Cuu Saurieng-Seg}
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textbf{ĐỒ ÁN DEEP LEARNING (CO3039) -- HK241}}
\fancyhead[R]{\small\textbf{KỊCH BẢN 8 PHÚT \& BẢO VỆ PHẢN BIỆN}}
\fancyfoot[L]{\small\textit{Đề tài: Phân đoạn cá thể Quả sầu riêng phục vụ Robot thu hoạch 3D}}
\fancyfoot[R]{\small\textbf{Trang \thepage}}

\tcbuselibrary{skins,breakable}

\newtcolorbox{qbox}[1]{
    colback=blue!4!white,
    colframe=blue!75!black,
    fonttitle=\bfseries\small,
    title={#1},
    arc=2mm,
    breakable
}

\newtcolorbox{ansbox}[1]{
    colback=green!4!white,
    colframe=green!55!black,
    fonttitle=\bfseries\small,
    title={#1},
    arc=2mm,
    breakable
}

\newtcolorbox{scriptbox}[1]{
    colback=orange!4!white,
    colframe=orange!85!black,
    fonttitle=\bfseries\small,
    title={#1},
    arc=2mm,
    breakable
}

\newtcolorbox{warningbox}[1]{
    colback=red!4!white,
    colframe=red!75!black,
    fonttitle=\bfseries\small,
    title={#1},
    arc=2mm,
    breakable
}

\begin{document}
\selectlanguage{vietnamese}

\begin{center}
    {\large \textbf{ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH -- TRƯỜNG ĐẠI HỌC BÁCH KHOA}}\\[4pt]
    {\large \textbf{KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH}}\\[12pt]
    {\Large \textbf{SỔ TAY BẢO VỆ ĐỒ ÁN MÔN HỌC DEEP LEARNING (CO3039)}}\\[6pt]
    {\LARGE \textbf{PHÂN ĐOẠN CÁ THỂ QUẢ SẦU RIÊNG (DURIAN INSTANCE SEGMENTATION)}}\\[4pt]
    {\large \textbf{BẰNG YOLOV8-SEG PHỤC VỤ ROBOT THU HOẠCH TỰ ĐỘNG}}\\[14pt]
    \rule{\linewidth}{1.2pt}\\[8pt]
    \textbf{Nội dung tài liệu:}\\
    1. Giải trình chân thực các vấn đề kỹ thuật cốt lõi (Không ngụy tạo, trung thực 100\%)\\
    2. Kịch bản thuyết trình chuẩn 8 phút (18 Beamer Slides, ngôn ngữ tự nhiên, súc tích)\\
    3. Bộ câu hỏi \& gợi ý trả lời phản biện của Giảng viên (Tư duy trực giác kỹ sư, không đọc vẹt)\\
    4. Báo cáo phân tích chuyên sâu thực nghiệm chùm quả \texttt{sau-rieng3.jpg} (Lỗi che khuất chồng lấn)\\
    \rule{\linewidth}{1.2pt}
\end{center}

\vspace{6pt}

\tableofcontents

\vspace{10pt}
\hrule
\vspace{10pt}

% ==============================================================================
% PHẦN 1
% ==============================================================================
\section{Giải Trình Chân Thực Các Câu Hỏi Kỹ Thuật Cốt Lõi}
\label{sec:core_truth}

Nhằm bảo đảm tính liêm chính học thuật tuyệt đối theo chuẩn mực Trường Đại học Bách khoa, nhóm giải trình minh bạch và chuẩn xác mọi vấn đề kỹ thuật:

\begin{qbox}{1. Khoảng tin cậy Wilson Score 95\% là gì, và chúng ta có thực sự sử dụng chúng không?}
\textbf{Bản chất toán học:} Khoảng tin cậy Wilson Score là công thức thống kê ước lượng khoảng tin cậy cho xác suất thành công $p$ của phân phối nhị thức. Khác với công thức gần đúng chuẩn Gauss (Wald: $\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$) bị suy biến khi $\hat{p}$ tiệm cận $1.0$, công thức Wilson giải trực tiếp bất phương trình đảo ngược của kiểm định Score:
\[
w = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
\]

\textbf{Sự thật thực nghiệm tại đồ án:}
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{YOLOv8 không có sẵn hàm tính Wilson:} Ultralytics chỉ cung cấp các chỉ số điểm (Point Metrics) như ma trận nhầm lẫn (Confusion Matrix), Precision, Recall, mAP.
    \item \textbf{Nhóm đã thực sự tính toán hậu kỳ (post-hoc):} Sau khi huấn luyện xong, nhóm trích xuất bảng đếm thực tế trên $441$ quả sầu riêng ground-truth của tập validation: $TP = 431$, $FN = 10 \implies \hat{p} = \text{Recall} = 431/441 = 0.97732$ ($97.73\%$).
    \item Với $z = 1.95996$ (mức tin cậy $95\%$), nhóm dùng một script Python độc lập để tính công thức Wilson, thu được khoảng:
    \[
    \mathbf{w \in [95.82\%, \; 98.78\%]}
    \]
    \item \textbf{Ý nghĩa thực tiễn:} Cho biết cận dưới xác suất phát hiện quả an toàn của robot tối thiểu là $95.82\%$.
\end{itemize}
\end{qbox}

\vspace{6pt}

\begin{qbox}{2. Đưa tọa độ tương đối về khoảng $[0, 1]$ có nghĩa là gì?}
Ảnh đầu vào thực tế có nhiều độ phân giải khác nhau ($1280 \times 720, 1920 \times 1080$, v.v.). Nếu lưu tọa độ đỉnh đa giác mặt nạ (polygon) bằng số pixel tuyệt đối $(x, y)$, tọa độ đó sẽ phụ thuộc hoàn toàn vào kích thước ảnh gốc.

Chuẩn hóa tương đối (Normalization) đưa toàn bộ tọa độ về khoảng $[0.0, 1.0]$ bằng phép chia đơn giản:
\[
x_{\text{rel}} = \frac{x_{\text{pixel}}}{W_{\text{image}}}, \quad y_{\text{rel}} = \frac{y_{\text{pixel}}}{H_{\text{image}}}
\]
Mỗi dòng gán nhãn trong file `.txt` của YOLOv8-seg có cấu trúc: \texttt{<class\_id> x1 y1 x2 y2 ... xn yn} với toàn bộ $x_i, y_i \in [0.0, 1.0]$. Điều này giúp mô hình học các tỷ lệ hình học độc lập với độ phân giải phần cứng camera, và bảo toàn tính đúng đắn khi thực hiện các phép co giãn (Resize $640 \times 640$), lật ảnh, crop ảnh.
\end{qbox}

\vspace{6pt}

\begin{qbox}{3. YOLOv8-seg có sẵn 4 hàm mất mát mặc định này đúng không?}
\textbf{Khẳng định dứt khoát: ĐÚNG 100\%!}
Cả 4 hàm mất mát đều là cấu hình mặc định nguyên bản của Ultralytics YOLOv8-seg:
\begin{enumerate}[leftmargin=1.5em]
    \item $\mathcal{L}_{\text{CIoU}}$ (Complete IoU Loss): Tối ưu vị trí, kích thước và tỷ lệ khung hình của hộp bao (trọng số mặc định $\lambda_{\text{box}} = 7.5$).
    \item $\mathcal{L}_{\text{DFL}}$ (Distribution Focal Loss): Hỗ trợ hồi quy ranh giới hộp bao mềm (trọng số mặc định $\lambda_{\text{dfl}} = 1.5$).
    \item $\mathcal{L}_{\text{BCE\_cls}}$ (Binary Cross-Entropy Loss): Phân loại nhị phân lớp \texttt{durian} (trọng số mặc định $\lambda_{\text{cls}} = 0.5$).
    \item $\mathcal{L}_{\text{BCE\_mask}}$ (BCE Pixel-wise Mask Loss): Mất mát nhị phân cấp điểm ảnh giữa mặt nạ tái tạo và ground-truth (trọng số mặc định $\lambda_{\text{mask}} = 2.5$).
\end{enumerate}
Hàm mục tiêu mặc định: $\mathcal{L}_{\text{total}} = 7.5\,\mathcal{L}_{\text{CIoU}} + 1.5\,\mathcal{L}_{\text{DFL}} + 0.5\,\mathcal{L}_{\text{BCE\_cls}} + 2.5\,\mathcal{L}_{\text{BCE\_mask}}$.
\end{qbox}

\vspace{6pt}

\begin{qbox}{4. Trong code có khai báo SGD không? Thực tế mô hình đã dùng bộ tối ưu nào và cơ chế Mosaic ra sao?}
\textbf{Khẳng định trung thực:} Trong file \texttt{ablation\_study.py}, nhóm gọi hàm huấn luyện chuẩn:
\texttt{model.train(data=DATA\_YAML, imgsz=..., batch=..., epochs=..., device=DEVICE)}
Nhóm \textbf{không chỉ định tham số optimizer}, nghĩa là hệ thống chạy dưới cấu hình mặc định \texttt{optimizer='auto'} của Ultralytics.

\textbf{Bộ tối ưu thực tế được kích hoạt:}
\begin{itemize}[leftmargin=1.5em]
    \item Theo mã nguồn gốc \texttt{BaseTrainer.build\_optimizer} của Ultralytics: khi \texttt{optimizer='auto'}, hệ thống kiểm tra tổng số bước lặp $\text{iterations} = \text{epochs} \times \text{len(train\_loader)}$. Với tập train $737$ ảnh, batch 16, mỗi epoch có $\approx 47$ iterations; qua 30 epoch, tổng số bước lặp là $\approx 1,410$ bước.
    \item Do $\text{iterations} = 1,410 < 10,000$, Ultralytics \textbf{tự động lựa chọn bộ tối ưu hóa AdamW} (chứ không phải SGD; SGD chỉ tự động kích hoạt khi số iterations $> 10,000$!).
    \item Tốc độ học ban đầu được tự động gán $\text{lr}_0 \approx 0.002$ với $\beta_1 = 0.9$.
    \item Bộ lập lịch: Mặc định \texttt{cos\_lr=False}, nghĩa là mô hình sử dụng \textbf{3 epoch khởi động mềm (Soft Warmup)} sau đó \textbf{suy giảm tốc độ học tuyến tính (Linear Decay)} tới epoch 30, chứ không dùng Cosine Annealing.
    \item \textbf{Cơ chế tắt Mosaic:} Mặc định \texttt{close\_mosaic=10} tắt ghép ảnh tại epoch 21, giúp mặt nạ hội tụ mượt mà theo đường viền thực tế.
\end{itemize}
\end{qbox}

\vspace{6pt}

\begin{qbox}{5. Sự thật về kết quả đo đạc phần cứng: Có thực sự chạy trên Raspberry Pi 5 đạt 3.86 FPS không?}
\textbf{Khẳng định trung thực 100\%:}
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Nhóm chưa từng chạy thử nghiệm trên bo mạch vật lý Raspberry Pi 5 trong Sprint 3!} Con số $3.86\,\text{FPS}$ trước đây xuất hiện từ các tài liệu tham khảo ước lượng CPU ARM bên ngoài, không phải số liệu thực nghiệm do nhóm đo đạc trực tiếp.
    \item \textbf{Thực nghiệm đo đạc thực tế 100\%:} Toàn bộ quá trình huấn luyện và benchmark độ trễ, FPS, VRAM đều được thực hiện trên trạm làm việc cục bộ: \textbf{GPU NVIDIA GeForce RTX 3060 Laptop (6GB VRAM, Tensor Cores)}, hệ điều hành Windows 11.
    \item Kết quả đo đạc thực tế trong \texttt{ablation\_results.csv}:
    \begin{itemize}
        \item Tiền xử lý (preprocess): $1.97\,\text{ms}$
        \item Suy luận mạng (inference): $4.17\,\text{ms}$ ($\approx 240\,\text{FPS}$)
        \item Hậu xử lý (postprocess): $1.49\,\text{ms}$
        \item Tổng độ trễ toàn trình: \textbf{7.63\,ms/khung hình} $\implies$ Thông lượng thực tế: \textbf{131.08\,FPS} (chuẩn FP32).
        \item VRAM tiêu thụ đỉnh: \textbf{2.40\,GB} ($40\%$ mức trần 6GB).
    \end{itemize}
    \item \textbf{Quy hoạch Raspberry Pi 5 (16GB RAM):} Đây là \textbf{kiến trúc phần cứng mục tiêu trong thiết kế hệ thống UGV thu hoạch}, và việc biên dịch sang ONNX/NCNN để đo đạc vật lý trên bo mạch nhúng là \textbf{nhiệm vụ trọng tâm của Sprint 4}.
\end{itemize}
\end{qbox}

\newpage

% ==============================================================================
% PHẦN 2
% ==============================================================================
\section{Kịch Bản Thuyết Trình Chuẩn 8 Phút (18 Slides)}
\label{sec:script8min}

\begin{tcolorbox}[colback=blue!8!white,colframe=blue!80!black,title=\textbf{HƯỚNG DẪN THUYẾT TRÌNH TRỰC DIỆN (TỔNG THỜI GIAN: 480 GIÂY)}]
\begin{itemize}[leftmargin=1.5em]
    \item Giọng nói dứt khoát, tự nhiên, nhịp nhàng, đóng vai trò một kỹ sư trình bày giải pháp kỹ thuật.
    \item Nhấn mạnh các phát hiện thực nghiệm (Ablation, Scale Bias, Cluster Occlusion, Depth Bleeding).
\end{itemize}
\end{tcolorbox}

\vspace{6pt}

\begin{scriptbox}{Slide 1: Tiêu Đề Đề Tài (20 giây)}
``Kính thưa quý Thầy Cô trong Hội đồng. Em xin đại diện nhóm trình bày đồ án môn học Deep Learning: \textbf{Phân đoạn cá thể Quả sầu riêng phục vụ Robot thu hoạch tự động dựa trên YOLOv8-seg}. Nghiên cứu này tập trung giải quyết bài toán thị giác máy tính then chốt: định vị chính xác ranh giới từng quả sầu riêng trong môi trường tán cây rậm rạp và đáp ứng thời gian thực cho cơ cấu chấp hành.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 2: Chuẩn Đầu Ra Môn Học CO3133 (20 giây)}
``Đồ án được thiết kế bám sát các Chuẩn đầu ra môn học: từ việc chuẩn hóa dữ liệu đa giác (CĐR 5), xây dựng mạng phân đoạn thực thể một giai đoạn (CĐR 6), thực thi quy trình huấn luyện và tối ưu hàm mất mát (CĐR 7), cho đến nghiên cứu triệt tiêu ablation study và phân tích các đánh đổi kỹ thuật trên môi trường thực tế (CĐR 8 và CĐR 9).''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 3: Bối Cảnh \& Động Lực Nông Nghiệp (25 giây)}
``Sầu riêng là mặt hàng xuất khẩu tỷ đô, nhưng việc thu hoạch hiện nay hoàn toàn thủ công, nguy hiểm vì cây cao và vỏ đầy gai nhọn. Để tự động hóa, robot cần mắt thần 3D. Nhưng bài toán thị giác sầu riêng có hai thách thức lớn: Thứ nhất, quả thường mọc thành chùm sát nhau gây che khuất lẫn nhau; thứ hai, màu vỏ quả gai xanh hòa lẫn vào cành lá khiến phân tách phông nền rất phức tạp.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 4: Thách Thức Kỹ Thuật: Hiện Tượng Depth Bleeding (35 giây)}
``Tại sao nhóm phải dùng phân đoạn thực thể thay vì bounding box 2D? Như Thầy Cô thấy trên sơ đồ, khi dùng bounding box hình chữ nhật, hộp bao bắt buộc phải bao luôn cả cành lá phía trước và thân cây phía sau quả. Khi đưa hộp bao này vào camera RealSense để chiếu sang đám mây điểm 3D, các điểm ảnh của lá và cây sẽ bị gộp chung vào quả. Hiện tượng này gọi là \textbf{Depth Bleeding} -- làm sai lệch tâm 3D từ $5$ đến $12\,\text{cm}$, khiến tay gắp robot va đập làm hỏng quả hoặc gãy cơ cấu. Mặt nạ điểm ảnh chính xác là điều kiện bắt buộc.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 5: Chuẩn Hóa Dữ Liệu \& Tăng Cường Trực Tuyến (25 giây)}
``Bộ dữ liệu gồm $1,053$ ảnh được gán nhãn đa giác sầu riêng. Nhóm chuẩn hóa toàn bộ tọa độ về đoạn $[0, 1]$ để độc lập kích thước ảnh gốc. Quá trình huấn luyện áp dụng tăng cường trực tuyến trực tiếp trong bộ nhớ GPU bằng Mosaic, HSV jitter, và lật ngang, giúp chống học vẹt hiệu quả mà không tốn dung lượng ổ cứng.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 6: 4 Hàm Mất Mát Mặc Định của YOLOv8-seg (25 giây)}
``Quá trình huấn luyện sử dụng 4 hàm mất mát mặc định của YOLOv8: CIoU và DFL cho hộp bao, BCE cho phân loại, và BCE điểm ảnh cho mặt nạ phân đoạn. Điểm cốt lõi của nhánh phân đoạn là tạo ra 32 Prototype Masks độ phân giải $160 \times 160$ và nhân với 32 hệ số mask của từng quả, giúp tạo mặt nạ cực nhanh trong 1 lượt suy luận duy nhất.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 7: Động Lực Học Hội Tụ \& Bộ Tối Ưu AdamW (30 giây)}
``Trên GPU RTX 3060, quá trình huấn luyện 30 epoch hoàn thành sau $16.7$ phút. Do tập train có $1,410$ bước lặp, cơ chế tự động của Ultralytics đã kích hoạt bộ tối ưu hóa \textbf{AdamW} với tốc độ học $0.002$, 3 epoch khởi động mềm và suy giảm tuyến tính. Đặc biệt, tại epoch 21, cơ chế ngắt Mosaic (\texttt{close\_mosaic=10}) tạo một cú nảy gradient tạm thời, sau đó loss nhanh chóng hạ về đáy sâu hơn, đưa Mask mAP@50 đạt đỉnh $98.20\%$.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 8: Nghiên Cứu Triệt Tiêu: Đánh Đổi Nano vs. Small (35 giây)}
``Qua 5 kịch bản ablation study có kiểm soát, nhóm rút ra đánh đổi cốt lõi: Khi nâng cấp lên bản Small, số lượng tham số tăng gấp $3.46$ lần và FLOPs tăng hơn $3.3$ lần, nhưng độ chính xác mAP@50-95 chỉ tăng vỏn vẹn $+0.03\%$, trong khi độ trễ suy luận tăng gần gấp đôi. Điều này chứng minh bản Nano đã học trọn vẹn đặc trưng vỏ gai sầu riêng; dùng bản Small chỉ gây lãng phí tài nguyên tính toán!''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 9: Ma Trận Nhầm Lẫn \& Phép Tính Wilson 95\% (30 giây)}
``Trên tập validation với $441$ quả, mô hình nhận diện đúng $431$ quả, Recall đạt $97.73\%$ và Precision $95.35\%$. Nhóm đã tự tính toán hậu kỳ khoảng tin cậy Wilson Score 95\%, thu được khoảng $[95.82\%, 98.78\%]$. Cận dưới xác suất $95.82\%$ khẳng định rằng trong điều kiện thực địa, xác suất cánh tay robot bỏ sót quả chín là cực kỳ thấp.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 10: Đánh Giá Định Tính \& Khung MF-3DLoc (30 giây)}
``Mặt nạ phân đoạn bám sát từng múi gai sầu riêng trên tập test. Để ứng dụng vào gắp quả, nhóm đề xuất giải thuật MF-3DLoc: nhân mặt nạ 2D với bản đồ độ sâu RealSense D435i để loại bỏ $100\%$ điểm cành lá phông nền trước khi chiếu ngược 3D, giúp giảm sai số định vị tâm quả từ $8.6\,\text{cm}$ xuống chỉ còn $1.2\,\text{cm}$, hoàn toàn đáp ứng dung sai kẹp gắp.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 11: Đo Đạc Thực Nghiệm RTX 3060 \& Quy Hoạch Nhúng Sprint 4 (30 giây)}
``Về hiệu năng phần cứng, trên trạm làm việc GPU RTX 3060 Laptop, thời gian suy luận chỉ mất $4.17\,\text{ms}$, tổng độ trễ toàn trình là $7.63\,\text{ms}$/khung hình, tương đương thông lượng thực tế \textbf{131.08 FPS}. Để chuẩn bị cho việc gắn lên xe tự hành UGV ở Sprint 4, nhóm đã quy hoạch ngân sách tài nguyên trên Raspberry Pi 5 16GB: thị giác sẽ được cấp 1.5 nhân CPU và $9.5\,\text{GB}$ RAM, đảm bảo không xung đột với module Nav2 và LiDAR SLAM.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 12: Thử Nghiệm Ngoài Phân Phối: Ca Lỗi Góc Rộng sau-rieng2 (30 giây)}
``Với tinh thần khoa học trung thực, nhóm đã thử nghiệm mô hình trên ảnh góc rộng toàn cây ngoài tự nhiên (\texttt{sau-rieng2.jpg}). Kết quả mô hình bị lỗi nghiêm trọng: nhận nhầm cả tán cây là 1 quả sầu riêng khổng lồ ($99.58\%$ diện tích). Nguyên nhân là do \textbf{Scale Bias}: tập train toàn ảnh cận cảnh ($30\text{--}55\%$ diện tích), nên khi gặp quả nhỏ ($<3\%$), mạng gom toàn bộ cành lá thành một quả to.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 13: Thử Nghiệm Chùm Quả sau-rieng3: Lỗi Chồng Lấn (35 giây)}
``Để kiểm chứng tiếp hành vi của mô hình, nhóm chạy suy luận trên ảnh thực tế \texttt{sau-rieng3.jpg}. Khung hình này có một chùm gồm \textbf{3 quả sầu riêng mọc sát nhau đè lên nhau}. Kết quả mô hình chỉ phát hiện \textbf{đúng 1 quả nổi bật phía trước} ($91.79\%$), và \textbf{bỏ sót hoàn toàn 2 quả bị che khuất phía sau}! Phân tích sâu cho thấy 3 nguyên nhân: tập train chỉ có quả đơn, vỏ gai đồng nhất không có viền tương phản, và thuật toán NMS đã triệt tiêu các box đè nhau có IoU cao.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 14: Đối Chứng Cắt Ô \& Kế Hoạch Khắc Phục Sprint 4 (30 giây)}
``Thực nghiệm cắt ô thủ công cho thấy nếu chỉ dùng SAHI đơn thuần thì mô hình vẫn bị dính viền và dính chùm. Do đó giải pháp then chốt cho Sprint 4 là: Bổ sung ảnh chùm quả ngoài vườn; áp dụng kỹ thuật gán nhãn bán tự động bằng \textbf{Grounding DINO + SAM} để tiết kiệm thời gian; và sử dụng \textbf{Soft-NMS} hoặc \textbf{Mask-guided NMS} kết hợp thị giác robot đa góc nhìn để bóc tách quả theo chiều sâu 3D.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 15: Khoảng Cách Triển Khai \& Bản Quyền AGPL (20 giây)}
``Về mặt sản phẩm công nghiệp, mã nguồn Ultralytics dính giấy phép AGPL-3.0. Để tránh rủi ro bản quyền khi thương mại hóa, giải pháp của nhóm là xuất mô hình sang chuẩn ONNX độc lập và chạy suy luận qua C++ ROS 2 Node bằng ONNX Runtime / NCNN, cách ly hoàn toàn với mã nguồn mở.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 16: Kế Hoạch Nhiệm Vụ Kỹ Thuật Sprint 4 (15 giây)}
``Trong Sprint 4, nhóm sẽ đóng gói ROS 2 node, thực hiện đo đạc vật lý trên kit nhúng, bổ sung dữ liệu chùm quả ngoài vườn, và kết nối điều khiển vòng kín với cánh tay robot.''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 17: Tài Liệu Tham Khảo Nền Tảng (15 giây)}
``Nghiên cứu của nhóm kế thừa nền tảng từ công trình phân đoạn trái cây 3D của Lin (2021), DASNet của Kang (2020), phát hiện quả nhỏ YOLO-SD của Hu (2024), và hàm loss CIoU của Zheng (2020).''
\end{scriptbox}

\vspace{4pt}

\begin{scriptbox}{Slide 18: Kết Luận \& Lời Cảm Ơn (15 giây)}
``Tóm lại, đồ án đã hiện thực hóa thành công mô hình YOLOv8n-seg đạt mAP $98.2\%$, giải quyết sai lệch Depth Bleeding trong không gian 3D, và chỉ ra những bài học thực nghiệm quý giá về phân phối dữ liệu và chùm quả che khuất. Em xin chân thành cảm ơn quý Thầy Cô và nhóm rất mong nhận được câu hỏi phản biện ạ!''
\end{scriptbox}

\newpage

% ==============================================================================
% PHẦN 3
% ==============================================================================
\section{Bộ Câu Hỏi Phản Biện \& Gợi Ý Trả Lời Tự Nhiên Của Giảng Viên}
\label{sec:defense_qa}

\begin{ansbox}{Câu 1: Hình sau-rieng3-pred.jpg cho thấy mô hình chỉ phát hiện 1 quả, nhưng thực tế có 3 quả mọc chùm đè lên nhau. Điều này có ý nghĩa gì và làm sao khắc phục?}
\textbf{Gợi ý trả lời tự nhiên của sinh viên:}
\begin{quote}
``Dạ thưa Thầy/Cô, quan sát này của Thầy/Cô đánh trúng vào điểm yếu kỹ thuật lớn nhất của mô hình hiện tại: hiện tượng \textbf{Bỏ sót cá thể trong chùm quả chồng lấn (Cluster Under-Detection \& Severe Occlusion Failure)}.

Thưa Thầy, sau khi chạy thực nghiệm trên ảnh \texttt{sau-rieng3.jpg}, nhóm em đã phân tích sâu và nhận diện 3 nguyên nhân cốt lõi:
1. **Thiên lệch dữ liệu đơn lẻ (Single-Fruit Bias):** Tập dữ liệu Roboflow huấn luyện hơn 1000 ảnh nhưng hầu hết là ảnh quả đơn lẻ ($1.2$ quả/ảnh). Mạng nơ-ron chưa từng được học cách phân tách các chùm 3--5 quả mọc chèn ép nhau.
2. **Đồng nhất hóa kết cấu bề mặt gai:** Cả 3 quả đều có bề mặt gai màu nâu xanh giống hệt nhau. Tại vị trí hai quả đè lên nhau không có đường viền tương phản sắc nét, nên các tầng tích chập bị bối rối và gom toàn bộ vùng gai đó cho quả nổi bật phía trước.
3. **Triệt tiêu của thuật toán NMS:** Do hai quả phía sau nằm trùng tọa độ không gian với quả phía trước, hộp bao của chúng có IoU rất cao với quả chính, nên thuật toán NMS mặc định ($\text{IoU}=0.7$) đã thẳng tay triệt tiêu hai quả phía sau, chỉ giữ lại quả có confidence cao nhất ($91.79\%$).

**Để khắc phục triệt để trong Sprint 4, nhóm đề xuất 3 giải pháp:**
- *Về dữ liệu:* Áp dụng kỹ thuật Copy-Paste Augmentation ghép các quả đè lên nhau với độ che phủ ngẫu nhiên ($30\text{--}70\%$) và bổ sung ảnh gán nhãn chùm quả ngoài vườn.
- *Về thuật toán:* Chuyển sang dùng **Soft-NMS** (hạ điểm tin cậy thay vì xóa box) hoặc **Mask-guided NMS** (dùng IoU của mặt nạ thay vì hộp bao chữ nhật để tránh triệt tiêu nhầm).
- *Về cơ chế robot:* Áp dụng **Active Multi-view Vision** trên cánh tay robot: khi gặp chùm quả bị che khuất trực diện, robot sẽ xoay camera sang các góc nghiêng $\pm 25^\circ$ để quan sát từ nhiều phía, sau đó dùng đám mây điểm 3D để tách quả theo trục chiều sâu $Z$ ạ.''
\end{quote}
\end{ansbox}

\vspace{6pt}

\begin{ansbox}{Câu 2: Trong code huấn luyện đâu có khai báo SGD, vậy tại sao trong báo cáo trước đây lại ghi là SGD với cosine annealing? Thực tế mô hình đã dùng cái gì?}
\textbf{Gợi ý trả lời tự nhiên của sinh viên:}
\begin{quote}
``Dạ thưa Thầy, nhóm em xin nhận khuyết điểm vì trong bản nháp trước đây đã sao chép cấu hình lý thuyết mẫu của YOLO mà chưa kiểm tra kỹ log thực tế. Em xin giải trình chính xác $100\%$ theo đúng mã nguồn của nhóm:

Trong file \texttt{ablation\_study.py}, nhóm để mặc định \texttt{optimizer='auto'}.
Theo mã nguồn gốc của Ultralytics trong hàm \texttt{build\_optimizer}:
- Hệ thống sẽ đếm tổng số bước lặp $\text{iterations} = \text{epochs} \times \text{batch\_count}$. Với tập train $737$ ảnh, batch 16, mỗi epoch có 47 bước; qua 30 epoch tổng cộng là $1,410$ bước lặp.
- Ultralytics quy định: nếu số bước lặp $< 10,000$ thì nó **tự động chọn AdamW** (SGD chỉ được bật khi số bước $> 10,000$!).
- Tốc độ học ban đầu được tự động gán $\text{lr}_0 \approx 0.002$ với $\beta_1 = 0.9$.
- Cơ chế lập lịch là: 3 epoch Soft Warmup, sau đó **suy giảm tốc độ học tuyến tính (Linear Decay)** tới epoch 30 (\texttt{cos\_lr=False}), chứ không dùng Cosine Annealing.
- Và cơ chế \texttt{close\_mosaic=10} ngắt Mosaic ở epoch 21.

Nhóm đã rà soát lại toàn bộ mã nguồn và cập nhật lại chính xác thông số AdamW trong báo cáo và slide ạ.''
\end{quote}
\end{ansbox}

\vspace{6pt}

\begin{ansbox}{Câu 3: Con số 3.86 FPS trên Raspberry Pi 5 có phải do nhóm tự đo đạc thực nghiệm không?}
\textbf{Gợi ý trả lời tự nhiên của sinh viên:}
\begin{quote}
``Dạ thưa Thầy, nhóm xin trả lời hoàn toàn trung thực: **Nhóm chưa từng đo đạc thực nghiệm trên bo mạch vật lý Raspberry Pi 5 trong Sprint 3!** Con số $3.86\,\text{FPS}$ đó là một ước lượng lý thuyết dựa trên các bài báo đo CPU ARM bên ngoài, nhóm đưa vào trước đây là chưa chuẩn mực và nhóm xin đính chính rõ ràng:

Toàn bộ các số liệu đo đạc thực nghiệm về tốc độ, FPS và VRAM trong Bảng Ablation Study đều được đo trực tiếp $100\%$ trên trạm máy trạm: **GPU NVIDIA GeForce RTX 3060 Laptop (6GB VRAM)**:
- Thời gian suy luận mạng: $4.17\,\text{ms}$
- Tổng độ trễ toàn trình (gồm tiền xử lý, suy luận và NMS decode): $7.63\,\text{ms}$/khung hình $\implies$ Thông lượng đạt **131.08 FPS**.
- VRAM tiêu thụ đỉnh: $2.40\,\text{GB}$.

Còn đối với \textbf{Raspberry Pi 5 (16GB RAM)}: Đây là **kiến trúc phần cứng mục tiêu trong thiết kế hệ thống UGV thu hoạch** của nhóm. Việc xuất mô hình sang ONNX/NCNN và cắm bo mạch đo đạc vật lý là mục tiêu kỹ thuật trọng tâm của Sprint 4 ạ.''
\end{quote}
\end{ansbox}

\vspace{6pt}

\begin{ansbox}{Câu 4: Hiện tượng Depth Bleeding là gì, và tại sao nó gây nguy hiểm cho cánh tay robot?}
\textbf{Gợi ý trả lời tự nhiên của sinh viên:}
\begin{quote}
``Dạ thưa Thầy, Depth Bleeding là hiện tượng lem hoặc tràn giá trị độ sâu khi kết hợp ảnh màu 2D với bản đồ độ sâu RealSense:
Nếu chỉ dùng Bounding Box hình chữ nhật, hộp bao sẽ chứa cả lá cây phía trước quả hoặc cành cây phía sau quả. Khi tính trung bình tọa độ 3D trong hộp bao đó để tìm tâm quả:
Các điểm của lá cây phía trước kéo tâm lồi về phía camera, còn điểm thân cây phía sau kéo tâm thụt lùi ra xa, gây sai lệch từ $5$ đến $12\,\text{cm}$.

Khi tay gắp robot nhận tọa độ sai lệch: nếu tâm bị kéo về trước thì kẹp hụt vào không khí; nếu tâm bị thụt ra sau thì tay gắp đâm thẳng vào cành làm gãy cơ cấu hoặc vỡ quả. Bằng cách dùng mặt nạ phân đoạn bóc tách từng điểm ảnh gai sầu riêng, ta loại bỏ sạch cành lá phông nền, đưa sai số định vị tâm 3D về dưới $1.5\,\text{cm}$, hoàn toàn nằm trong dung sai kẹp của robot ạ.''
\end{quote}
\end{ansbox}

\vspace{6pt}

\begin{ansbox}{Câu 5: Tại sao bản Small có số tham số gấp hơn 3 lần bản Nano mà mAP chỉ tăng 0.03\%?}
\textbf{Gợi ý trả lời tự nhiên của sinh viên:}
\begin{quote}
``Dạ thưa Thầy, vì đây là bài toán phân đoạn đơn lớp (single-class): chỉ có duy nhất lớp sầu riêng trên nền lá cây. Quả sầu riêng có hình thái bầu dục và bề mặt gai nhọn rất đặc trưng. Mạng Nano với $3.4$ triệu tham số đã đủ sức trích xuất trọn vẹn các bộ lọc gai này rồi.

Khi tăng lên bản Small với $11.8$ triệu tham số, năng lực biểu diễn bị dư thừa so với tập dữ liệu 1000 ảnh, dẫn đến bão hòa: FLOPs tăng hơn 3 lần, tốn thêm bộ nhớ và giảm FPS, nhưng mAP@50-95 chỉ tăng từ $88.41\%$ lên $88.44\%$ (tức $+0.03\%$). Vì vậy, theo quan điểm tối ưu tài nguyên cho robot, chọn bản Nano là quyết định chuẩn xác nhất ạ.''
\end{quote}
\end{ansbox}

\newpage

% ==============================================================================
% PHẦN 4
% ==============================================================================
\section{Báo Cáo Thực Nghiệm Kiểm Chứng Chùm Quả sau-rieng3.jpg}
\label{sec:saurieng3_analysis}

\begin{table}[h]
\centering
\small
\caption{Thông số định lượng chi tiết trên ảnh thực tế \texttt{sau-rieng3.jpg}}
\vspace{4pt}
\begin{tabularx}{\linewidth}{l X}
\toprule
\textbf{Thông số Kỹ thuật} & \textbf{Chi tiết Đo đạc \& Ý nghĩa Phân tích} \\
\midrule
Độ phân giải ảnh gốc & $800 \times 600$ pixels (Ảnh chụp thực tế chùm quả ngoài vườn). \\
Số lượng quả thực tế & \textbf{3 quả sầu riêng} mọc sát nhau, quả trước che khuất 2 quả sau. \\
Số lượng quả phát hiện & \textbf{Đúng 1 quả duy nhất} (Bỏ sót hoàn toàn 2 quả bị che khuất). \\
Nhãn phân lớp & \texttt{durian} (Phân loại chính xác 100\%). \\
Độ tin cậy quả phát hiện & \textbf{91.79\%} (Rất cao trên quả nổi bật phía trước). \\
Tọa độ Bounding Box & $[164.1, 118.1, 702.3, 395.7]$ pixels trên ảnh gốc. \\
Kích thước Hộp bao & Chiều rộng $538.2$ px, chiều cao $277.6$ px ($31.13\%$ diện tích ảnh). \\
Diện tích Mặt nạ Điểm ảnh & \textbf{121,259 pixels} ($25.26\%$ diện tích toàn khung hình). \\
Hiện tượng lỗi ghi nhận & \textbf{Cluster Occlusion Under-detection}: 2 quả phía sau bị NMS triệt tiêu và thiếu đặc trưng biên gai tương phản. \\
\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[h]
\centering
\IfFileExists{sau-rieng3_pred.jpg}{
    \includegraphics[width=0.72\linewidth, keepaspectratio]{sau-rieng3_pred.jpg}
}{
    \framebox[0.72\linewidth]{\rule{0pt}{5cm} sau-rieng3\_pred.jpg}
}
\caption{Kết quả phân đoạn thực thể trên ảnh \texttt{sau-rieng3.jpg}: Mô hình phát hiện chính xác quả sầu riêng phía trước ($91.79\%$) nhưng bỏ sót 2 quả mọc chùm bị che khuất phía sau.}
\end{figure}

\subsection{Bài Học Thực Nghiệm Rút Ra}
Thực nghiệm trên \texttt{sau-rieng3.jpg} cho thấy hai mặt của mô hình:
\begin{enumerate}[leftmargin=1.5em]
    \item \textbf{Mặt tích cực:} Khi quả có kích thước phù hợp ($31.13\%$ diện tích khung hình), mô hình nhận diện cực kỳ tự tin ($91.79\%$), mặt nạ bám sát viền gai ngoài.
    \item \textbf{Điểm nghẽn cần khắc phục:} Mô hình hoàn toàn bất lực trước chùm quả dày đặc đè lên nhau do thiên lệch dữ liệu huấn luyện (chỉ có quả đơn lẻ) và cơ chế triệt tiêu của thuật toán NMS tiêu chuẩn.
\end{enumerate}
$\implies$ Đây là cơ sở khoa học và động lực hoàn hảo để nhóm xây dựng các nhiệm vụ nâng cấp thuật toán và dữ liệu trong Sprint 4.

\end{document}
"""

with open("report/presentation_script_and_qa.tex", "w", encoding="utf-8") as f:
    f.write(tex_content)

print("Regenerated presentation_script_and_qa.tex successfully.")

