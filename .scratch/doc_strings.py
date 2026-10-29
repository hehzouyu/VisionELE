import re, sys

def extract(path):
    data = open(path, 'rb').read()
    s = data.decode('utf-16-le', errors='ignore')
    segs = re.split(r'[^\x20-\x7e\u3000-\u303f\u4e00-\u9fff\uff01-\uff5e]{2,}', s)
    out = []
    for seg in segs:
        seg = seg.strip()
        if len(seg) >= 8 and not seg.startswith('Microsoft') and 'Word.Document' not in seg:
            out.append(seg)
    return '\n'.join(out)

if __name__ == '__main__':
    sys.stdout.buffer.write(extract(sys.argv[1]).encode('utf-8', errors='replace'))
