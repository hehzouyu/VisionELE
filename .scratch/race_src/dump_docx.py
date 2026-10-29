import sys, os
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

def iter_block(parent):
    from docx.oxml.ns import qn
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)

def dump(path, out):
    doc = Document(path)
    lines = []
    for block in iter_block(doc):
        if isinstance(block, Paragraph):
            t = block.text.strip()
            if not t:
                continue
            style = (block.style.name or '')
            lvl = ''
            if style.startswith('Heading') or style.startswith('标题'):
                lvl = f'[{style}] '
            lines.append(lvl + t)
        else:
            lines.append('<<TABLE>>')
            for row in block.rows:
                cells = [' '.join(c.text.split()) for c in row.cells]
                lines.append(' | '.join(cells))
            lines.append('<<END TABLE>>')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'{path} -> {out} ({len(lines)} lines)')

if __name__ == '__main__':
    dump(sys.argv[1], sys.argv[2])
