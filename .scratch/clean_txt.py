import glob, os
d = r'E:\创新创业\.scratch\lit_src'
ok = 0
for p in sorted(glob.glob(os.path.join(d, '*.txt'))):
    if p.endswith('l1.txt') or p.endswith('l2.txt'):
        continue
    with open(p, 'rb') as f:
        b = f.read()
    try:
        s = b.decode('utf-8')
        bad = 0
    except UnicodeDecodeError:
        s = b.decode('utf-8', errors='replace')
        bad = s.count('\ufffd')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(s)
    ok += 1
print('cleaned', ok)
