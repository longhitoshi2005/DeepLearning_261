import re
import os

print("--- 1. UPDATING main_vi.tex ---")
with open("report/main_vi.tex", "r", encoding="utf-8") as f:
    vi_text = f.read()

# Fix SGD to AdamW in main_vi.tex
old_sgd_vi = "giúp bộ tối ưu hóa SGD tái hiệu chuẩn các bộ lọc tích chập"
new_adamw_vi = "giúp bộ tối ưu hóa AdamW (được cơ chế tự động của Ultralytics lựa chọn cho 1,410 bước lặp) tái hiệu chuẩn các bộ lọc tích chập"
vi_text = vi_text.replace(old_sgd_vi, new_adamw_vi)

# Make sure Raspberry Pi 5 is clearly introduced as target architecture planning, not physical measured benchmark in Sprint 3
old_pi_intro = r"""Do máy tính nhúng Raspberry Pi 5 vận hành bằng nguồn pin trên bo mạch robot và không trang bị GPU rời công suất lớn như trạm máy tính để bàn, bộ vi xử lý trung tâm 4 nhân ARM Cortex-A76 phải gánh vác đồng thời toàn bộ các phân hệ tự hành then chốt:"""
new_pi_intro = r"""Cần lưu ý rằng trong khuôn khổ Sprint 3, toàn bộ các số liệu đo đạc thực nghiệm về độ trễ, FPS và VRAM (Bảng~\ref{tab:ablation_results}) đều được đo đạc trực tiếp 100\% trên trạm làm việc GPU NVIDIA GeForce RTX 3060 Laptop (6GB VRAM) với thông lượng toàn trình đạt $131.08\,\text{FPS}$ ($7.63\,\text{ms}$/khung hình). Tuy nhiên, để chuẩn bị cho việc đóng gói và chuyển giao lên robot thực địa ở Sprint 4, nhóm đã tiến hành phân tích và quy hoạch ngân sách tài nguyên trên kiến trúc phần cứng mục tiêu: máy tính nhúng \textbf{Raspberry Pi 5 (16\,GB RAM)} kết hợp camera RealSense D453i và LiDAR C1M1. Do Raspberry Pi 5 vận hành bằng nguồn pin và không có GPU rời, bộ vi xử lý 4 nhân ARM Cortex-A76 sẽ phải gánh vác đồng thời toàn bộ các phân hệ tự hành:"""
vi_text = vi_text.replace(old_pi_intro, new_pi_intro)

# Add sau-rieng3 analysis into main_vi.tex
sau_rieng3_vi = r"""
\subsubsection{Ca kiểm chứng chùm quả \texttt{sau-rieng3.jpg}: Hiện tượng bỏ sót cá thể do che khuất và chồng lấn}
Để kiểm chứng phản ứng của mô hình khi gặp chùm quả ngoài tự nhiên, nhóm đã chạy suy luận trên ảnh thực tế \texttt{sau-rieng3.jpg}. Khung hình này chứa một chùm gồm \textbf{3 quả sầu riêng mọc sát nhau và nằm đè chồng lấn lên nhau}:
\begin{itemize}
    \item \textbf{Kết quả thực nghiệm:} Mô hình chỉ phát hiện được \textbf{đúng 1 quả nổi bật nhất ở phía trước} với độ tin cậy $91.79\%$ (hộp bao $[164.1, 118.1, 702.3, 395.7]$ chiếm $31.13\%$ diện tích ảnh), trong khi \textbf{2 quả sầu riêng mọc phía sau bị bỏ sót hoàn toàn (False Negatives)}!
    \item \textbf{Bản chất và Nguyên nhân gốc rễ:}
    \begin{enumerate}
        \item \textit{Thiên lệch dữ liệu đơn lẻ (Single-Fruit Bias):} Tập dữ liệu Roboflow huấn luyện gồm 1,053 ảnh hầu hết là các quả đơn lẻ mọc biệt lập (trung bình chỉ $1.2$ quả/ảnh). Mạng nơ-ron chưa từng được học cách phân tách các chùm quả dày đặc mọc đè lên nhau.
        \item \textit{Đồng nhất hóa kết cấu bề mặt gai (Texture Homogenization):} Cả 3 quả đều có bề mặt gai màu nâu xanh tương đồng. Tại ranh giới tiếp xúc giữa hai quả không có độ tương phản màu sắc lớn, khiến các tầng tích chập coi cụm gai liền kề là thuộc về quả lớn phía trước.
        \item \textit{Triệt tiêu của thuật toán NMS (Non-Maximum Suppression):} Do hai quả phía sau nằm trùng vị trí không gian với quả phía trước, các đề xuất hộp bao của chúng có IoU cao với quả chính, dẫn đến việc bị thuật toán NMS mặc định ($\text{IoU} = 0.7$) triệt tiêu.
    \end{enumerate}
    \item \textbf{Giải pháp cho Sprint 4:} Áp dụng kỹ thuật tăng cường tổng hợp Copy-Paste đè quả với tỷ lệ ngẫu nhiên ($30\text{--}70\%$), chuyển sang thuật toán \textbf{Soft-NMS} hoặc \textbf{Mask-guided NMS} (dựa trên IoU mặt nạ thay vì hộp bao), và áp dụng cơ chế thị giác chủ động \textbf{Active Multi-view Vision} trên cánh tay robot (xoay camera $\pm 25^\circ$ để bóc tách quả bằng đám mây điểm chiều sâu 3D).
\end{itemize}
"""

if "sau-rieng3.jpg" not in vi_text:
    # Insert right before 4.4.2
    target_pos = vi_text.find(r"\subsubsection{Khoảng cách triển khai và Hiện tượng biến dạng ranh giới}")
    if target_pos != -1:
        vi_text = vi_text[:target_pos] + sau_rieng3_vi + "\n" + vi_text[target_pos:]
    else:
        # append before subsection 4.5
        target_pos = vi_text.find(r"\subsection{Phân tích bản quyền phần mềm}")
        vi_text = vi_text[:target_pos] + sau_rieng3_vi + "\n" + vi_text[target_pos:]

with open("report/main_vi.tex", "w", encoding="utf-8") as f:
    f.write(vi_text)
print("Updated main_vi.tex successfully.")

print("--- 2. UPDATING main.tex ---")
with open("report/main.tex", "r", encoding="utf-8") as f:
    en_text = f.read()

old_sgd_en = "allows the SGD optimizer to re-calibrate"
new_adamw_en = "allows the AdamW optimizer (automatically selected by Ultralytics for 1,410 iterations) to re-calibrate"
en_text = en_text.replace(old_sgd_en, new_adamw_en)

old_pi_intro_en = r"""Because the Raspberry Pi 5 operates under battery power on the mobile robot without a dedicated discrete desktop GPU, its 4 ARM Cortex-A76 CPU cores must support the entire autonomous navigation and manipulation stack concurrently:"""
new_pi_intro_en = r"""It is important to emphasize that in Sprint 3, all quantitative latency, FPS, and VRAM benchmarks (Table~\ref{tab:ablation_results}) were physically measured 100\% on the local NVIDIA GeForce RTX 3060 Laptop GPU workstation, achieving an end-to-end throughput of $131.08\,\text{FPS}$ ($7.63\,\text{ms}$/frame). To prepare for physical robot deployment in Sprint 4, the team conducted an architectural resource budgeting analysis for the target mobile hardware: an embedded \textbf{Raspberry Pi 5 (16\,GB RAM)} coupled with an Intel RealSense D453i camera and LiDAR C1M1. Operating without a discrete GPU, its 4 ARM Cortex-A76 CPU cores must concurrently support the entire robotics stack:"""
en_text = en_text.replace(old_pi_intro_en, new_pi_intro_en)

sau_rieng3_en = r"""
\subsubsection{Cluster Occlusion Case Study on \texttt{sau-rieng3.jpg}: Missed Detections in Overlapping Fruits}
To evaluate model behavior when encountering dense fruit clusters in natural orchards, inference was executed on \texttt{sau-rieng3.jpg}. This real orchard scene contains a cluster of \textbf{3 durians growing in close proximity and heavily overlapping each other}:
\begin{itemize}
    \item \textbf{Empirical Observation:} The model successfully detects \textbf{only 1 dominant foreground durian} with $91.79\%$ confidence (BBox $[164.1, 118.1, 702.3, 395.7]$ covering $31.13\%$ of the frame), while the \textbf{2 occluded durians behind it are completely missed (False Negatives)}!
    \item \textbf{Root Cause Analysis:}
    \begin{enumerate}
        \item \textit{Single-Fruit Dataset Bias:} The Roboflow training set (1,053 images) contains predominantly isolated fruits (mean $1.2$ fruits/image). The convolutional backbone was never exposed to dense clusters with heavy occlusion.
        \item \textit{Spike Texture Homogenization:} All fruits share identical greenish-brown spiky textures. At overlapping contact boundaries, the absence of sharp contrast borders causes convolutional feature maps to assimilate the cluster into the dominant foreground instance.
        \item \textit{Non-Maximum Suppression (NMS) Elimination:} Candidate proposals for the background fruits exhibit high spatial IoU overlap with the foreground fruit's bounding box, resulting in their suppression by standard NMS ($\text{IoU} = 0.7$).
    \end{enumerate}
    \item \textbf{Mitigation Strategy for Sprint 4:} Incorporating synthetic Copy-Paste overlap augmentation ($30\text{--}70\%$ occlusion), adopting \textbf{Soft-NMS} or \textbf{Mask-guided NMS} (using mask IoU rather than bounding box IoU to avoid spurious suppression), and deploying \textbf{Active Multi-view Vision} on the robot manipulator ($\pm 25^\circ$ viewpoint shifting to segment fruits in 3D depth point clouds).
\end{itemize}
"""

if "sau-rieng3.jpg" not in en_text:
    target_pos = en_text.find(r"\subsubsection{Production Deployment Gap and Boundary Artifacts}")
    if target_pos != -1:
        en_text = en_text[:target_pos] + sau_rieng3_en + "\n" + en_text[target_pos:]
    else:
        target_pos = en_text.find(r"\subsection{Open-Source Licensing Compliance}")
        en_text = en_text[:target_pos] + sau_rieng3_en + "\n" + en_text[target_pos:]

with open("report/main.tex", "w", encoding="utf-8") as f:
    f.write(en_text)
print("Updated main.tex successfully.")

