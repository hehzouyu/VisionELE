# -*- coding: utf-8 -*-
"""用标准库把 docx 的 document.xml 转成纯文本（保留段落与表格行的粗结构）。"""
import zipfile
import re
import os

BASE = r"E:\创新创业\国创"
OUT = r"E:\创新创业\.scratch\guochuang"
os.makedirs(OUT, exist_ok=True)

files = {
    "plan_CA_robot": "基于CA注意力机制的绝缘子缺陷检测机器人.docx",
    "plan_P020260": "P020260416537636563556.docx",
    "tech_vision": "技术设想.docx",
}

for name, fn in files.items():
    path = os.path.join(BASE, fn)
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    # 行内换行、单元格、段落、表格行
    xml = xml.replace("</w:tc>", "\t")
    xml = xml.replace("</w:tr>", "\n")
    xml = xml.replace("</w:br>", "\n")
    xml = xml.replace("</w:p>", "\n")
    text = re.sub(r"<[^>]+>", "", xml)
    # html 实体
    for a, b in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")]:
        text = text.replace(a, b)
    text = re.sub(r"\n{2,}", "\n", text)
    out = os.path.join(OUT, name + ".txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{name}: {len(text)} chars -> {out}")
