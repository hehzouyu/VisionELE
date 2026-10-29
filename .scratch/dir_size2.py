import os, subprocess, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
root = r"E:\创新创业"
base = r"E:\创新创业\Demos\insulator_guardian"
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
    parts = n.split("/")
    key = "/".join(parts[:3]) if len(parts) > 2 else n
    agg[key] += s
    cnt[key] += 1
for k, v in agg.most_common(18):
    print(f"{v/1024/1024:9.2f} MB  {cnt[k]:5d}  {k}")
