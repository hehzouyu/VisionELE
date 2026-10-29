import os, subprocess, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
root = r"E:\创新创业"
out = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True)
names = [n for n in out.stdout.decode("utf-8", "surrogateescape").split("\0") if n]
agg = collections.Counter()
cnt = collections.Counter()
for n in names:
    p = os.path.join(root, n)
    try:
        s = os.path.getsize(p)
    except OSError:
        continue
    top = "/".join(n.split("/")[:2])
    agg[top] += s
    cnt[top] += 1
for k, v in agg.most_common(15):
    print(f"{v/1024/1024:9.2f} MB  {cnt[k]:5d}  {k}")
