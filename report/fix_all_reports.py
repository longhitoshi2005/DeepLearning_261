import os, re

def fix_file(file_path, is_vi=True):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if is_vi:
        # Fix Figure 3.1 Caption
        old_cap = re.search(r'\\caption\{Phân tích thực nghiệm ca lỗi suy biến quy mô.*?\\label\{fig:scale_failure\}', text, re.DOTALL)
        new_cap = r'''\caption{Phân tích thực nghiệm ca lỗi suy biến quy mô và overfitting trên ảnh vườn tự nhiên (\texttt{sau-rieng2.jpg}). (a) Mô hình nhận diện nhầm toàn bộ tán cây thành 1 quả sầu riêng lớn trùm kín $99.58\%$ diện tích ảnh ($conf = 92.56\%$). (b--c) Cắt ô cụm quả 1: mô hình nhận diện sai lệch khi gộp dính 2 quả làm 1 ($conf = 74.8\%$) và bỏ sót hoàn toàn quả thứ 3 bên phải. (d--e) Cắt ô vùng quả 2: mô hình dự đoán hộp bao trùm kín gần $100\%$ ô cắt ($conf = 82.9\%$), mặt nạ bị rò rỉ nghiêm trọng (boundary bleeding) lên cả thân cây và cành lá xung quanh.}
\label{fig:scale_failure}'''
        if old_cap:
            text = text[:old_cap.start()] + new_cap + text[old_cap.end():]

        # Fix Section 3.4.1 body
        p1 = text.find(r'\paragraph{Phân tích kiểm chứng đối chứng cắt ô')
        if p1 == -1:
            p1 = text.find(r'\paragraph{Kiểm chứng giải pháp đối chứng')
        p2 = text.find(r'\subsubsection{Các hạn chế phần cứng và pháp lý khác}')

        body_vi = r'''\paragraph{Phân tích kiểm chứng đối chứng cắt ô: Bộc lộ bản chất Overfitting sâu sắc:}
Khi tiến hành cắt các ô nhỏ xung quanh vùng quả trên ảnh \texttt{sau-rieng2.jpg} nhằm đưa kích thước quả về tỷ lệ tương đồng với tập huấn luyện ($20\%\text{--}30\%$ khung hình), kết quả thực nghiệm tại Hình~\ref{fig:scale_failure}(b--e) đã bộc lộ rõ ràng các khiếm khuyết cốt lõi của mô hình:
\begin{itemize}
    \item \textbf{Tại ô cắt chùm quả 1 (Hình~\ref{fig:scale_failure}c):} Vùng cắt thực tế có 3 quả sầu riêng (chùm 2 quả bên trái và 1 quả đơn bên phải). Mô hình chỉ dự đoán một hộp bao duy nhất gộp dính cả 2 quả bên trái thành một đối tượng (\texttt{durian 0.75}), đồng thời bỏ sót hoàn toàn quả thứ 3 bên phải (False Negative). Điều này chứng minh mô hình bị suy thoái năng lực phân đoạn thực thể riêng biệt (under-segmentation / instance clumping) khi các quả mọc san sát nhau trong chùm tự nhiên.
    \item \textbf{Tại ô cắt quả 2 (Hình~\ref{fig:scale_failure}e):} Quả sầu riêng nằm xen giữa tán lá và thân cây. Mô hình dự đoán một hộp bao trùm kín gần $100\%$ toàn bộ ô cắt (\texttt{durian 0.83}), kéo theo mặt nạ phân đoạn rò rỉ nghiêm trọng (boundary bleeding) lên cả thân cây và lá cây xung quanh. Hiện tượng này khẳng định mạng bị thiên kiến ``chiếm trọn khung hình'' (crop-filling bias): cứ đưa ô nào vào là mạng ép dự đoán một hộp bao choán hết diện tích ô đó mà không phân biệt được ranh giới thực giữa vỏ quả và vỏ cây.
\end{itemize}
Những phát hiện thực nghiệm này khẳng định rằng: \textbf{Mô hình hiện tại đã bị overfitting nặng vào tập dữ liệu ảnh chụp cận cảnh đơn lẻ}, và kỹ thuật cắt ô SAHI thuần túy sẽ không thể khắc phục triệt để nếu mô hình nền tảng chưa được học các đặc trưng trong môi trường tự nhiên thực tế.

\noindent\textbf{Định hướng khắc phục triệt để cho Sprint 4:}
\begin{enumerate}
    \item \textbf{Thu thập bộ dữ liệu thực địa ngoài tự nhiên (In-situ Orchard Dataset):} Bổ sung tối thiểu 300--500 ảnh chụp toàn cảnh tán cây và cự ly trung bình (chụp tại vườn sầu riêng thực tế hoặc nguồn tư liệu thực địa mở) với đa dạng cự ly ($0.5\text{--}5.0\,\text{m}$), mật độ chùm quả phức tạp, quả bị che khuất và điều kiện ánh sáng tự nhiên khác nhau.
    \item \textbf{Giải quyết bài toán gán nhãn tự động bằng Foundation Model (Grounded-SAM):} Nhằm loại bỏ rào cản gán nhãn đa giác thủ công tốn kém hàng trăm giờ công, nhóm sẽ ứng dụng quy trình gán nhãn bán tự động tiên tiến: Sử dụng mô hình nhận diện zero-shot \textbf{Grounding DINO} \cite{liu2023grounding} kết hợp với mô hình nền móng phân đoạn \textbf{Segment Anything (SAM / SAM 2)} \cite{kirillov2023segment}. Văn bản gợi ý (text prompt \texttt{``durian fruit''}) sẽ định hướng Grounding DINO trích xuất bounding box thô, sau đó SAM tự động mở rộng và ôm khít viền gai để sinh nhãn đa giác chuẩn xác cấp pixel. Con người chỉ đóng vai trò kiểm toán và xác nhận nhanh (Human-in-the-loop verification), giúp tăng năng suất tạo dữ liệu lên gấp $20\text{--}30$ lần mà không cần vẽ thủ công từng đỉnh.
    \item \textbf{Tăng cường dữ liệu Copy-Paste và Fine-tune trên SAHI:} Áp dụng kỹ thuật \textit{Instanced Scale Copy-Paste} để dán ngẫu nhiên các quả sầu riêng ở các tỷ lệ nhỏ ($0.05\text{--}0.25$) vào các ảnh nền cây thực tế, sau đó tái huấn luyện mô hình YOLOv8n-seg kết hợp thuật toán suy luận đa tỷ lệ SAHI \cite{akyon2022sahi}.
\end{enumerate}

'''
        if p1 != -1 and p2 != -1:
            text = text[:p1] + body_vi + text[p2:]

    else:
        # Fix English
        old_cap = re.search(r'\\caption\{Empirical investigation of scale invariance failure.*?\\label\{fig:scale_failure\}', text, re.DOTALL)
        new_cap = r'''\caption{Empirical investigation of scale invariance failure and contextual overfitting on wide-angle canopy imagery (\texttt{sau-rieng2.jpg}). (a) The baseline model misclassifies the entire tree canopy as a single giant durian occupying $99.58\%$ of the image ($conf = 92.56\%$) due to extreme training scale bias. (b--c) Fruit cluster 1 patch: the model exhibits under-segmentation, clumping two adjacent fruits into a single entity ($conf = 74.8\%$) and completely missing the third fruit. (d--e) Fruit 2 patch: the model exhibits crop-filling bias ($conf = 82.9\%$), where the bounding box spans nearly $100\%$ of the patch and the mask bleeds severely onto the surrounding tree trunk and foliage.}
\label{fig:scale_failure}'''
        if old_cap:
            text = text[:old_cap.start()] + new_cap + text[old_cap.end():]

        p1 = text.find(r'\paragraph{Counter-Validation via Local Patch Slicing')
        p2 = text.find(r'\subsubsection{Physical and Hardware Constraints}')

        body_en = r'''\paragraph{Counter-Validation via Local Patch Slicing: Exposing Severe Contextual Overfitting:}
When local crops were extracted around fruit regions in \texttt{sau-rieng2.jpg} to restore the fruit area to proportions familiar from the training distribution ($20\%\text{--}30\%$ of the patch), the empirical results in Figure~\ref{fig:scale_failure}(b--e) uncovered critical underlying failure modes:
\begin{itemize}
    \item \textbf{Cluster 1 Crop (Figure~\ref{fig:scale_failure}c):} The cropped region contains three fruits (a clumped pair on the left and a separate fruit on the right). The model predicted a single bounding box merging the two clumped fruits into one single entity (\texttt{durian 0.75}), while completely omitting the third fruit (False Negative). This demonstrates an instance under-segmentation and clumping failure when fruits grow in natural dense clusters.
    \item \textbf{Fruit 2 Crop (Figure~\ref{fig:scale_failure}e):} The target fruit is suspended between leaves and the tree trunk. The model predicted a bounding box occupying almost $100\%$ of the entire cropped patch (\texttt{durian 0.83}), causing severe boundary bleeding where the segmentation mask spilled broadly across the tree trunk and surrounding leaves. This reveals a chronic ``crop-filling bias'': whenever a sub-patch is ingested, the network presumes a single object spans the whole window rather than resolving the true morphological boundaries of the fruit.
\end{itemize}
These empirical observations prove that: \textbf{The current model suffers from severe contextual overfitting to close-up single-fruit images}, and relying solely on post-hoc slicing algorithms (SAHI) without retraining on natural in-situ orchard data cannot resolve these fundamental boundary and instance-separation defects.

\noindent\textbf{Comprehensive Mitigation Roadmap for Sprint 4:}
\begin{enumerate}
    \item \textbf{In-situ Orchard Dataset Collection:} Collect 300--500 high-resolution full-canopy and mid-range images directly in natural durian orchards across varying illumination conditions, camera distances ($0.5\text{--}5.0\,\text{m}$), and heavy cluster occlusions.
    \item \textbf{Automated Polygon Labeling via Foundation Models (Grounded-SAM):} To circumvent the bottleneck of manual polygon annotation (which requires hundreds of manual vertex clicking hours), we implement a zero-shot semi-automated annotation pipeline: Combining \textbf{Grounding DINO} \cite{liu2023grounding} with \textbf{Segment Anything (SAM / SAM 2)} \cite{kirillov2023segment}. High-level text prompts (\texttt{``durian fruit''}) direct Grounding DINO to propose rough bounding boxes, which are instantly ingested by SAM as prompt boxes to extract sub-pixel precision polygon contours. Human annotators act exclusively as rapid quality assurance auditors (Human-in-the-loop verification), boosting annotation throughput by $20\text{--}30\times$.
    \item \textbf{Scale-Jittered Copy-Paste Augmentation and SAHI Fine-Tuning:} Synthesize diverse fruit scales by randomly blending segmented fruits at small scale ratios ($0.05\text{--}0.25$) onto complex orchard canopy backgrounds, followed by fine-tuning YOLOv8n-seg alongside the SAHI inference framework \cite{akyon2022sahi}.
\end{enumerate}

'''
        if p1 != -1 and p2 != -1:
            text = text[:p1] + body_en + text[p2:]

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Fixed {file_path} successfully!')

fix_file('main_vi.tex' if os.path.exists('main_vi.tex') else 'report/main_vi.tex', is_vi=True)
fix_file('main.tex' if os.path.exists('main.tex') else 'report/main.tex', is_vi=False)
