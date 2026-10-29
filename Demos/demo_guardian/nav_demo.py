"""绝缘卫士 · 决策导航仿真 demo
A* 全局规划 + DWA 局部避障 + 缺陷事件触发重规划
运行: /d/conda/python nav_demo.py   -> 生成 nav_demo.gif
"""
import heapq
import math
import sys
import numpy as np
import matplotlib
if "--live" not in sys.argv:          # 生成 GIF 用无头后端；--live 时弹窗实时播放
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# ---------- 参数 ----------
G = 25                      # 栅格尺寸
DT = 0.1                    # 仿真步长(s)
V_MAX = 2.2                 # 最大线速度(格/s)
W_MAX = 2.0                 # 最大角速度(rad/s)
ROBOT_R = 0.8               # 机器人半径(格)

# ---------- 地图 ----------
grid = np.zeros((G, G), dtype=np.uint8)
for i in range(G):
    grid[0, i] = grid[G - 1, i] = grid[i, 0] = grid[i, G - 1] = 1

obs = [(3, 6), (4, 6), (5, 6), (6, 6), (6, 7), (6, 8), (6, 9),
       (15, 3), (16, 3), (17, 3), (18, 3), (18, 4), (18, 5),
       (10, 14), (11, 14), (12, 14), (13, 14), (13, 15), (13, 16),
       (20, 10), (21, 10), (22, 10), (22, 11), (22, 12),
       (8, 19), (9, 19), (9, 20), (9, 21)]
for a, b in obs:
    grid[a, b] = 1

# 机器人半径膨胀
inflated = grid.copy()
for i in range(G):
    for j in range(G):
        if grid[i, j]:
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    if di * di + dj * dj <= (ROBOT_R + 0.15) ** 2:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < G and 0 <= nj < G:
                            inflated[ni, nj] = 1

# ---------- A* ----------
DIRS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]

def astar(start, goal, gridmap):
    def h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    openq = [(h(start, goal), 0.0, start, None)]
    came, g = {}, {}
    while openq:
        _, gc, cur, parent = heapq.heappop(openq)
        if cur in came:
            continue
        came[cur] = parent
        if cur == goal:
            path, c = [], cur
            while c is not None:
                path.append(c)
                c = came[c]
            return path[::-1]
        for dx, dy, cost in DIRS:
            nx, ny = cur[0] + dx, cur[1] + dy
            if 0 <= nx < G and 0 <= ny < G and gridmap[nx, ny] == 0:
                ng = gc + cost
                if ng < g.get((nx, ny), 1e9):
                    g[(nx, ny)] = ng
                    heapq.heappush(openq, (ng + h((nx, ny), goal), ng, (nx, ny), cur))
    return None

def simplify(path):
    if len(path) <= 2:
        return path
    out = [path[0]]
    for i in range(1, len(path) - 1):
        prev, cur, nxt = out[-1], path[i], path[i + 1]
        if (cur[0] - prev[0]) * (nxt[1] - cur[1]) != (cur[1] - prev[1]) * (nxt[0] - cur[0]):
            out.append(cur)
    out.append(path[-1])
    return out

def line_clear(a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 6) + 1
    for t in np.linspace(0, 1, n):
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < G and 0 <= yi < G) or inflated[xi, yi] == 1:
            return False
    return True

def simplify_safe(path):
    """只在两段直线无碰撞时才跳过中间点，保留必要拐点"""
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_clear(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out

def densify(path, step=0.5):
    """把稀疏拐点路径插值成密集点，防止局部跟踪切角"""
    out = [path[0]]
    for a, b in zip(path, path[1:]):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for k in range(1, n + 1):
            out.append((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
    return out

# ---------- 局部避障：DWA 采样 ----------
def lookahead(path, pos, dist=1.0):
    """从最近路径点之后向前看，绝不回头指向身后"""
    k = min(range(len(path)), key=lambda i: math.hypot(path[i][0] - pos[0], path[i][1] - pos[1]))
    i = k + 1
    while i < len(path) and math.hypot(path[i][0] - pos[0], path[i][1] - pos[1]) < dist:
        i += 1
    return path[min(i, len(path) - 1)]

def trajectory_ok(x, y, th, v, w):
    th2, x2, y2 = th, x, y
    for _ in range(int(1.5 / DT)):
        th2 += w * DT
        x2 += v * DT * math.cos(th2)
        y2 += v * DT * math.sin(th2)
        xi, yi = int(round(x2)), int(round(y2))
        if not (0 <= xi < G and 0 <= yi < G) or inflated[xi, yi] == 1:
            return False
    return True

def dwa(pos, path):
    lx, ly = lookahead(path, pos)
    best = None
    for v in np.arange(0.0, V_MAX + 1e-9, 0.4):
        for w in np.arange(-W_MAX, W_MAX + 1e-9, 0.2):   # 含 0，且步长更细
            if not trajectory_ok(pos[0], pos[1], pos[2], v, w):
                continue
            th2 = pos[2]
            x2, y2 = pos[0], pos[1]
            for _ in range(int(1.5 / DT)):
                th2 += w * DT
                x2 += v * DT * math.cos(th2)
                y2 += v * DT * math.sin(th2)
            heading = math.atan2(ly - y2, lx - x2)
            dth = (heading - th2 + math.pi) % (2 * math.pi) - math.pi
            dist = math.hypot(lx - x2, ly - y2)
            score = 2.0 * (-abs(dth)) - 1.2 * dist + 0.6 * v - 0.15 * abs(w)
            if best is None or score > best[0]:
                best = (score, v, w)
    if best is None:
        return 0.0, 0.0
    return best[1], best[2]

# ---------- 任务与事件 ----------
start = (2.0, 2.0, 0.0)
inspect_pts = [(20, 4), (20, 20), (4, 20), (12, 12)]
defect_pt = (8.0, 10.0)

pos = list(start)
trails = [tuple(pos[:2])]
states = []                 # 每步状态：x, y, th, trail 长度, 缺陷是否已触发
target_idx = 0
path = None
defect_done = False
defect_target = False
defect_pending = False
post_inspect_1_left = False
MAX_STEPS = 2600

for step in range(MAX_STEPS):
    # 缺陷事件：离开 (20,20) 前往下一个巡检点时触发复检
    if (not defect_done and post_inspect_1_left and math.hypot(pos[0] - 20, pos[1] - 20) < 1.6):
        defect_pending = True
        post_inspect_1_left = False

    # 到点后先处理缺陷复检，再继续巡检
    if defect_pending:
        defect_pending = False
        defect_done = True
        defect_target = True
        path = None

    goal = None
    if defect_target:
        goal = defect_pt
    elif target_idx < len(inspect_pts):
        goal = inspect_pts[target_idx]
    else:
        break

    if path is None:
        raw = astar((int(round(pos[0])), int(round(pos[1]))), goal, inflated)
        path = densify(simplify_safe(raw)) if raw else None
        if path is None:
            break

    if math.hypot(pos[0] - goal[0], pos[1] - goal[1]) < 1.0:
        if defect_target:
            defect_target = False
            defect_done = True          # 复检完成，继续当前巡检点（不推进 target_idx）
        else:
            if target_idx == 1:
                post_inspect_1_left = True
            target_idx += 1
        path = None
        continue

    v, w = dwa(pos, path)
    if v == 0.0 and w == 0.0:
        path = None
        continue
    pos[2] += w * DT
    pos[0] += v * DT * math.cos(pos[2])
    pos[1] += v * DT * math.sin(pos[2])
    trails.append((pos[0], pos[1]))
    states.append((pos[0], pos[1], pos[2], len(trails), defect_target))

    if not (0 <= pos[0] < G and 0 <= pos[1] < G):
        break

print(f"steps={step}, reached={target_idx}, defect_revisit={'YES' if defect_done else 'NO'}")

# 全程均匀抽 240 帧（首尾都在）
n = len(states)
idx = np.linspace(0, n - 1, 240).astype(int)
frames = [states[i] for i in idx]

# ---------- 绘制动画 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=110)
ax.set_xlim(0, G)
ax.set_ylim(0, G)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_title("绝缘卫士 · 决策导航仿真（A* + DWA + 缺陷重规划）", fontsize=13, pad=10)
ax.set_xlabel("X (格)")
ax.set_ylabel("Y (格)")
ax.grid(True, linestyle="--", alpha=0.25)

obs_xy = np.argwhere(grid == 1)
ax.scatter(obs_xy[:, 0] + 0.5, obs_xy[:, 1] + 0.5, s=60, c="#3a3a4a", marker="s", zorder=3, label="障碍")
for p in inspect_pts:
    ax.plot(p[0], p[1], "o", ms=9, mfc="#2e7d32", mec="white", mew=1.2, zorder=4)
ax.plot(defect_pt[0], defect_pt[1], "X", ms=13, c="#d32f2f", zorder=5, label="缺陷复检点")
ax.plot(start[0], start[1], "o", ms=12, mfc="#1565c0", mec="white", mew=1.5, zorder=5, label="起点")
ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

trail_line, = ax.plot([], [], "-", c="#1565c0", lw=1.4, alpha=0.75, zorder=6)
robot = Circle((start[0], start[1]), ROBOT_R, fc="#ffb300", ec="#e65100", lw=1.5, zorder=7)
ax.add_patch(robot)
head, = ax.plot([], [], "-", c="#e65100", lw=2.0, zorder=8)
status = ax.text(0.5, G - 0.6, "", fontsize=9, ha="left", color="#444")

def update(i):
    x, y, th, tl, dd = frames[i]
    trail = trails[:tl]
    robot.center = (x, y)
    trail_line.set_data([p[0] for p in trail], [p[1] for p in trail])
    head.set_data([x, x + ROBOT_R * 1.5 * math.cos(th)], [y, y + ROBOT_R * 1.5 * math.sin(th)])
    txt = f"帧 {i + 1}/{len(frames)}   x={x:.1f}  y={y:.1f}  θ={math.degrees(th):.0f}°"
    if dd:
        txt += "   [正在前往缺陷复检点]"
    status.set_text(txt)
    return trail_line, robot, head, status

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False)
if "--live" in sys.argv:
    plt.show()                        # 实时窗口播放（Ctrl+C / 关窗口退出）
else:
    out = "nav_demo.gif"
    ani.save(out, writer=animation.PillowWriter(fps=25))
    print("saved:", out)
