import os

with open("report/slides.tex", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Slide 7: AdamW, Linear Decay, RTX 3060 Laptop GPU
slide7_target = r"""    \begin{block}{Diễn biến Hội tụ qua 30 Epochs}
    \scriptsize
    \begin{itemize}
        \item Huấn luyện trên \textbf{RTX 3060 (12\,GB)}, tổng thời gian: \textbf{26.85 phút}.
        \item Val loss giảm đơn điệu, mAP@50 đạt đỉnh \textbf{0.9820}.
    \end{itemize}
    \end{block}"""

slide7_replacement = r"""    \begin{block}{Chiến Lược Tối Ưu Hóa \& Hội Tụ}
    \scriptsize
    \begin{itemize}\setlength{\itemsep}{1pt}
        \item \textbf{Bộ tối ưu:} \textbf{AdamW} (do \texttt{optimizer='auto'} tự động kích hoạt cho 1,410 bước lặp, $\beta_1=0.9$, lr $\approx 0.002$).
        \item \textbf{Bộ lập lịch:} 3 epoch Soft Warmup, suy giảm tuyến tính (Linear Decay, \texttt{cos\_lr=False}).
        \item \textbf{Phần cứng thực nghiệm:} Huấn luyện trên GPU NVIDIA RTX 3060 Laptop (6GB VRAM), hoàn thành sau $16.7$ phút.
        \item Val loss giảm đơn điệu, Mask mAP@50 đạt đỉnh \textbf{0.9820}.
    \end{itemize}
    \end{block}"""

text = text.replace(slide7_target, slide7_replacement)

# 2. Update Slide 11: Real RTX 3060 Measurements vs Sprint 4 Pi 5 Planning
slide11_target = r"""% ------------------------------------------------------------------------------
% SLIDE 11: RÀNG BUỘC PHẦN CỨNG RASPBERRY PI 5
% ------------------------------------------------------------------------------
\begin{frame}{Ràng Buộc Biên Phần Cứng: Triển Khai Trên Raspberry Pi 5 (16 GB)}
\scriptsize
Hệ thống tính toán trung tâm chạy trên \textbf{Raspberry Pi 5 (16\,GB RAM)}, kết hợp \textbf{RealSense D453i} và \textbf{LiDAR C1M1}:
\vspace{0.05cm}
\begin{table}[htbp]
\centering
\scriptsize
\renewcommand{\arraystretch}{0.85}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{3.8cm} c c >{\raggedright\arraybackslash}p{2.8cm} X}
\toprule
\textbf{Tiến trình / Phân hệ} & \textbf{Tần số} & \textbf{RAM} & \textbf{CPU (4 nhân ARM)} & \textbf{Mức độ ưu tiên} \\
\midrule
LiDAR C1M1 SLAM \& Odometry & 12\,Hz & 2.0\,GB & 1 nhân (Core 0) & An toàn tự hành (Bắt buộc) \\
ROS 2 Nav2 Costmaps \& Planner & 10\,Hz & 2.0\,GB & 1 nhân (Core 1) & Tránh chướng ngại (Bắt buộc) \\
Động học tháp pháo \& PID & 50\,Hz & 0.5\,GB & 0.5 nhân (Core 2) & Bám mục tiêu ngắm bắn \\
Hệ điều hành, ROS 2 DDS & -- & 2.0\,GB & Nền tảng chia sẻ & Duy trì ổn định hệ thống \\
\midrule
\rowcolor{bkblue!15} \textbf{Ngân sách khả dụng cho Thị giác} & -- & \textbf{$\approx$ 9.5\,GB} & \textbf{1.5 nhân (Core 2+3)} & Mức trần tài nguyên cho phép \\
\midrule
\textit{Ứng viên A: YOLOv8s-seg} & 5--6\,FPS & $\approx 1.8$\,GB & 100\% CPU (Bão hòa), $\Delta T = +28^\circ$C & Trễ cao, nguy cơ quá nhiệt \\
\rowcolor{darkgreen!15} \textit{\textbf{Ứng viên B: YOLOv8n-seg}} & \textbf{15--20\,FPS} & \textbf{$\approx 1.1$\,GB} & \textbf{60\% CPU, $\Delta T = +11^\circ$C} & \textbf{Lựa chọn tối ưu toàn diện} \\
\bottomrule
\end{tabularx}
\end{table}
\vspace{-2pt}
\begin{alertblock}{Kết luận kỹ thuật về bài toán Compute-Bound}
\scriptsize
Bản 16\,GB RAM giải tỏa hoàn toàn nguy cơ tràn RAM. Tuy nhiên CPU là điểm nghẽn: bản Small ($40.34\,\text{GFLOPs}$) làm nghẽn CPU và quá nhiệt. \textbf{YOLOv8n-seg} tối ưu qua NCNN với tập lệnh SIMD ARM NEON là cấu hình khả thi duy nhất đạt $15\text{--}20\,\text{FPS}$ ổn định.
\end{alertblock}
\end{frame}"""

slide11_replacement = r"""% ------------------------------------------------------------------------------
% SLIDE 11: ĐO ĐẠC THỰC NGHIỆM RTX 3060 & QUY HOẠCH PHẦN CỨNG NHÚNG SPRINT 4
% ------------------------------------------------------------------------------
\begin{frame}{Đo Đạc Thực Nghiệm RTX 3060 \& Quy Hoạch Phần Cứng Nhúng Sprint 4}
\begin{columns}[T]
\begin{column}{0.48\textwidth}
    \begin{block}{Thực Đo 100\% trên GPU RTX 3060 Laptop}
    \scriptsize
    Đo đạc trực tiếp trên GPU máy trạm NVIDIA GeForce RTX 3060 Laptop (6GB VRAM, chế độ FP32):
    \begin{itemize}\setlength{\itemsep}{1pt}
        \item \textbf{Tiền xử lý (Preprocess):} $1.97\,\text{ms}$
        \item \textbf{Suy luận mạng (Inference):} $4.17\,\text{ms}$ ($\approx 240\,\text{FPS}$)
        \item \textbf{Hậu xử lý (Postprocess):} $1.49\,\text{ms}$
        \item \textbf{Tổng độ trễ toàn chu trình:} \textbf{7.63\,ms/khung hình}
        \item \textbf{Thông lượng thực tế:} \textbf{131.08\,FPS}
        \item \textbf{VRAM tiêu thụ đỉnh:} \textbf{2.40\,GB} ($40\%$ mức trần)
    \end{itemize}
    $\rightarrow$ Tốc độ đáp ứng vượt trội trên trạm điều hành.
    \end{block}
\end{column}
\begin{column}{0.50\textwidth}
    \begin{exampleblock}{Quy Hoạch Ngân Sách Cho Sprint 4 (Pi 5 16GB)}
    \scriptsize
    Phân bổ tài nguyên lý thuyết cho UGV tự hành:
    \begin{itemize}\setlength{\itemsep}{1pt}
        \item \textbf{LiDAR SLAM \& Odometry:} 1 nhân CPU, $2.0\,\text{GB}$ RAM.
        \item \textbf{ROS 2 Nav2 Planner:} 1 nhân CPU, $2.0\,\text{GB}$ RAM.
        \item \textbf{PID động học tháp pháo:} 0.5 nhân CPU, $0.5\,\text{GB}$ RAM.
        \item \textbf{Hệ điều hành \& ROS 2 DDS:} $2.0\,\text{GB}$ RAM.
        \item \textbf{Ngân sách khả dụng cho AI:} \textbf{1.5 nhân CPU, $\approx 9.5\,\text{GB}$ RAM}.
    \end{itemize}
    \textit{*Trung thực học thuật: Đo đạc thực tế trên kit Raspberry Pi 5 / Jetson là nhiệm vụ triển khai của Sprint 4.}
    \end{exampleblock}
\end{column}
\end{columns}
\end{frame}"""

text = text.replace(slide11_target, slide11_replacement)

# 3. Add Slide for sau-rieng3 cluster occlusion analysis right after Slide 12
slide13_new = r"""% ------------------------------------------------------------------------------
% SLIDE 13: THỬ NGHIỆM CHÙM QUẢ SAU-RIENG3: LỖI CHỒNG LẤN (CLUSTER OCCLUSION)
% ------------------------------------------------------------------------------
\begin{frame}{Thực Nghiệm Chùm Quả sau-rieng3: Lỗi Bỏ Sót Cá Thể Do Chồng Lấn}
\begin{columns}[c]
\begin{column}{0.47\textwidth}
    \centering
    \IfFileExists{sau-rieng3_pred.jpg}{
        \includegraphics[width=0.98\linewidth, height=0.62\textheight, keepaspectratio]{sau-rieng3_pred.jpg}
    }{
        \framebox[0.98\linewidth]{\rule{0pt}{4.0cm} sau-rieng3\_pred.jpg}
    }
    \vspace{1pt}
    \scriptsize Thực tế có \textbf{3 quả mọc chùm đè lên nhau}, nhưng mô hình chỉ phát hiện \textbf{đúng 1 quả} (Conf $91.79\%$, BBox $31.13\%$), bỏ sót 2 quả bị che khuất!
\end{column}
\begin{column}{0.51\textwidth}
    \begin{alertblock}{Bản Chất: Cluster Occlusion \& Under-Detection}
    \scriptsize
    \begin{itemize}\setlength{\itemsep}{1pt}
        \item \textbf{Thiên lệch tập train:} Dataset Roboflow chỉ có quả đơn ($1.2$ quả/ảnh), mạng chưa từng học chùm quả đè nhau.
        \item \textbf{Đồng nhất kết cấu gai:} Vỏ gai cùng màu nâu xanh, không có viền tương phản cao để phân tách 2 quả tiếp xúc.
        \item \textbf{Triệt tiêu NMS:} Hộp bao 2 quả phía sau bị thuật toán NMS ($\text{IoU}=0.7$) loại bỏ vì trùng vị trí với quả trước.
    \end{itemize}
    \end{alertblock}
    \begin{exampleblock}{Giải Pháp Khắc Phục Cho Sprint 4}
    \scriptsize
    \begin{itemize}\setlength{\itemsep}{1pt}
        \item \textbf{Data Augmentation:} Copy-Paste ghép quả đè nhau ngẫu nhiên ($30\text{--}70\%$) và gán nhãn chuyên biệt chùm quả.
        \item \textbf{Soft-NMS / Mask-guided NMS:} Triệt tiêu dựa trên IoU mặt nạ thay vì hộp bao chữ nhật để tránh xóa nhầm.
        \item \textbf{Active Multi-view Vision:} Robot xoay camera $\pm 25^\circ$ kết hợp tách quả trên đám mây điểm 3D.
    \end{itemize}
    \end{exampleblock}
\end{column}
\end{columns}
\end{frame}
"""

# Insert slide13_new right before Slide 13 (which becomes Slide 14)
target_slide13 = r"""% ------------------------------------------------------------------------------
% SLIDE 13: ĐỐI CHỨNG CẮT Ô VÀ ĐỀ XUẤT GIẢI PHÁP SPRINT 4
% ------------------------------------------------------------------------------"""

text = text.replace(target_slide13, slide13_new + "\n" + target_slide13)

with open("report/slides.tex", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated report/slides.tex successfully.")

