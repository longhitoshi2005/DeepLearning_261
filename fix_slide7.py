with open('report/slides.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start = -1
end = -1
for i, line in enumerate(lines):
    if "begin{block}{Chiến Lược Tối Ưu Hóa" in line:
        start = i
    if start != -1 and "end{alertblock}" in line and i > start:
        end = i
        break

if start != -1 and end != -1:
    new_block = [
        "    \\begin{block}{Chiến Lược Tối Ưu Hóa \\& Hội Tụ}\n",
        "    \\scriptsize\n",
        "    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n",
        "        \\item \\textbf{Tối ưu:} \\textbf{AdamW} (auto cho $1,410$ iter, $\\beta_1=0.9$, lr $\\approx 0.002$).\n",
        "        \\item \\textbf{Lập lịch:} 3 epoch Warmup mềm, suy giảm tuyến tính (Linear Decay).\n",
        "        \\item \\textbf{Thực nghiệm:} GPU RTX 3060 Laptop (6GB VRAM), train 30 epoch trong $16.7$ phút.\n",
        "        \\item Val loss giảm đều, Mask mAP@50 đạt đỉnh \\textbf{0.9820}.\n",
        "    \\end{itemize}\n",
        "    \\end{block}\n",
        "    \\vspace{-3pt}\n",
        "    \\begin{alertblock}{Tại sao tắt Mosaic ở 10 Epoch cuối (\\texttt{close\\_mosaic=10})?}\n",
        "    \\scriptsize\n",
        "    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n",
        "        \\item Mosaic ghép 4 ảnh giúp chống học vẹt ở 20 epoch đầu.\n",
        "        \\item 10 epoch cuối tắt Mosaic để mạng tinh chỉnh trên ảnh tự nhiên toàn vẹn.\n",
        "        \\item Đổi phân phối tạo \\textbf{cú nảy gradient tạm thời} tại Epoch 20, sau đó loss hạ sâu hơn.\n",
        "    \\end{itemize}\n",
        "    \\end{alertblock}\n"
    ]
    lines[start:end+1] = new_block
    with open('report/slides.tex', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Slide 7 tightened successfully.")
else:
    print(f"Indices not found: start={start}, end={end}")

