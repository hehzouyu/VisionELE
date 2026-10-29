import glob, os
d = r'E:\创新创业\.scratch\lit_src'
files = [p for p in sorted(glob.glob(os.path.join(d, '*.txt'))) if not p.endswith(('l1.txt','l2.txt'))]
for i, p in enumerate(files):
    with open(p, encoding='utf-8') as f:
        lines = f.readlines()
    out = os.path.join(d, 'digest_%02d.md' % i)
    with open(out, 'w', encoding='utf-8') as g:
        g.write('# ' + os.path.basename(p) + '\n\n')
        g.write(''.join(lines[:90]))
print('done', len(files))
