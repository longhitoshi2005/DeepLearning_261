import sys, os

slide_path = 'slides.tex' if os.path.exists('slides.tex') else 'report/slides.tex'
with open(slide_path, 'r', encoding='utf-8') as f:
    text = f.read()

s12 = text.find('SLIDE 12: PHÁT HIỆN THỰC NGHIỆM')
s14 = text.find('SLIDE 14: KHOẢNG CÁCH TRIỂN KHAI')

slide12_13 = """SLIDE 12: PHÁT HIỆN THỰC NGHIỆM THEN CHỐT: CA LỖI SUY BIẾN QUY MÔ
% ------------------------------------------------------------------------------
\\begin{frame}{Phát Hiện Thực Nghiệm: Ca Lỗi Suy Biến Quy Mô \\& Overfitting}
\\begin{columns}[c]
\\begin{column}{0.54\\textwidth}
    \\centering
    \\IfFileExists{scale_failure_analysis.png}{
        \\includegraphics[width=0.98\\linewidth, height=0.72\\textheight, keepaspectratio]{scale_failure_analysis.png}
    }{
        \\framebox[0.98\\linewidth]{\\rule{0pt}{4.5cm} scale\\_failure\\_analysis.png}
    }
\\end{column}
\\begin{column}{0.44\\textwidth}
    \\begin{alertblock}{Bóc tách 3 Dạng Sai Lệch Thực Nghiệm}
    \\scriptsize
    \\begin{itemize}
        \\item \\textbf{(a) Toàn cây góc rộng:} Nhận nhầm cả tán cây là 1 quả sầu riêng ($99.58\\%$ ảnh, conf $92.56\\%$).
        \\item \\textbf{(c) Cắt chùm quả 1:} Gộp dính 2 quả làm 1 đối tượng (conf $74.8\\%$), bỏ sót hoàn toàn quả thứ 3 (FN)!
        \\item \\textbf{(e) Cắt vùng quả 2:} Bbox chiếm gần $100\\%$ ô cắt (conf $82.9\\%$), nạ tràn viền lên thân và lá!
    \\end{itemize}
    \\end{alertblock}
    \\vspace{-4pt}
    \\begin{block}{Kiểm toán 3 Nguyên nhân Gốc rễ}
    \\scriptsize
    \\begin{enumerate}
        \\item \\textbf{Thiên kiến quy mô:} Train set $100\\%$ cận cảnh ($1.00$ quả/ảnh, chiếm $30\\text{--}55\\%$ diện tích).
        \\item \\textbf{Overfitting video:} Học thuộc lòng kết cấu xanh rêu phủ kín khung hình.
        \\item \\textbf{Ảo tưởng mAP Validation:} mAP $0.9820$ che giấu hoàn toàn lỗi ngoài phân phối (OOD).
    \\end{enumerate}
    \\end{block}
\\end{column}
\\end{columns}
\\end{frame}

% ------------------------------------------------------------------------------
% SLIDE 13: ĐỐI CHỨNG CẮT Ô VÀ BẢN CHẤT OVERFITTING
% ------------------------------------------------------------------------------
\\begin{frame}{Thực Nghiệm Đối Chứng Cắt Ô \\& Đề Xuất Giải Pháp Triệt Để}
\\begin{columns}[T]
\\begin{column}{0.48\\textwidth}
    \\begin{alertblock}{Bản chất Khiếm khuyết của Mô hình}
    \\scriptsize
    \\begin{itemize}
        \\item \\textbf{Hội chứng chiếm trọn khung hình (Crop-filling bias):} Cứ đưa ô cắt nào vào là mạng ép dự đoán một hộp bao choán hết diện tích ô đó.
        \\item \\textbf{Suy thoái phân đoạn chùm:} Không thể tách rời các quả mọc san sát nhau trong chùm tự nhiên.
        \\item \\textbf{Rò rỉ ngữ nghĩa sang thân cây:} Chưa học được đặc trưng phân biệt giữa gai quả và vỏ thân sần sùi.
        \\item $\\rightarrow$ \\textbf{Kết luận:} Mô hình bị \\textbf{overfitting nặng}. SAHI thuần túy không thể giải quyết nếu chưa có dữ liệu tự nhiên!
    \\end{itemize}
    \\end{alertblock}
\\end{column}
\\begin{column}{0.5\\textwidth}
    \\begin{exampleblock}{Định Hướng Khắc Phục Triệt Để (Sprint 4)}
    \\scriptsize
    \\begin{enumerate}
        \\item \\textbf{Thu thập Dataset vườn thực địa (In-situ):} Bổ sung 300--500 ảnh toàn cây ngoài trời với đa dạng cự ly ($0.5\\text{--}5.0\\,\\text{m}$) và ánh sáng.
        \\item \\textbf{Gán nhãn tự động Foundation Models:}
        \\begin{itemize}
            \\scriptsize
            \\item Kết hợp \\textbf{Grounding DINO} \\cite{liu2023grounding} (Text prompt $\\rightarrow$ BBox thô) + \\textbf{SAM / SAM 2} \\cite{kirillov2023segment} (Tự động ôm viền gai sinh polygon mask).
            \\item Con người chỉ duyệt kiểm toán (Human-in-the-loop), giải quyết nút thắt gán nhãn thủ công (nhanh gấp $20\\text{--}30\\times$).
        \\end{itemize}
        \\item \\textbf{Tăng cường Scale Copy-Paste \\& SAHI:} Trộn đa quy mô quả lên nền cây và suy luận SAHI \\cite{akyon2022sahi}.
    \\end{enumerate}
    \\end{exampleblock}
\\end{column}
\\end{columns}
\\end{frame}

% ------------------------------------------------------------------------------
% """

text = text[:s12] + slide12_13 + text[s14:]

with open(slide_path, 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated slides.tex successfully via script!')
