"""
人造卫星轨道衰减模拟
基于 NRLMSISE-00 标准大气数据表 + RK4 数值积分
可视化风格：单窗口实时轨道动画
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tkinter import Tk, simpledialog
import sys

# ── NRLMSISE-00 标准大气密度数据表 ─────────────────────────────────────────────
_TABLE = np.array([
    [  0, 1.225e+00], [ 10, 4.135e-01], [ 20, 8.891e-02], [ 30, 1.841e-02],
    [ 40, 3.996e-03], [ 50, 1.027e-03], [ 60, 3.097e-04], [ 70, 8.283e-05],
    [ 80, 1.846e-05], [ 90, 3.416e-06], [100, 5.297e-07], [110, 9.648e-08],
    [120, 2.438e-08], [130, 8.484e-09], [140, 3.845e-09], [150, 2.070e-09],
    [160, 1.233e-09], [180, 5.194e-10], [200, 2.541e-10], [220, 1.367e-10],
    [250, 6.073e-11], [300, 1.916e-11], [350, 7.014e-12], [400, 2.803e-12],
    [450, 1.184e-12], [500, 5.215e-13], [600, 1.137e-13], [700, 2.818e-14],
    [800, 7.998e-15], [900, 2.490e-15], [1000, 8.510e-16],
])
_h_km   = _TABLE[:, 0]
_log_rho = np.log(_TABLE[:, 1])

def get_density(h_m):
    h_km = float(np.clip(h_m / 1e3, _h_km[0], _h_km[-1]))
    return float(np.exp(np.interp(h_km, _h_km, _log_rho)))

# ── 物理常数 ───────────────────────────────────────────────────────────────────
G   = 6.674e-11
M   = 5.972e24
R_E = 6371e3
Cd  = 2.2
A   = 10.0
m   = 500.0
dt  = 10.0        # 积分步长 s
H_STOP = 100e3    # 再入高度 m

# ── tkinter 弹窗：选择初始高度 ─────────────────────────────────────────────────
def choose_height():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    while True:
        res = simpledialog.askstring(
            "轨道参数设置",
            "请输入数字选择初始轨道高度：\n\n"
            "【1】300 km  — 较低轨道，衰减快（~10 天）\n"
            "【2】400 km  — 国际空间站高度（~93 天）\n"
            "【3】500 km  — 较高轨道，衰减慢（~数年）\n\n"
            "提示：空格键 暂停/继续，+/- 调节速度",
            parent=root,
        )
        if res in ("1", "2", "3"):
            root.destroy()
            return int(res)
        elif res is None:
            root.destroy()
            sys.exit()

choice = choose_height()
H0_MAP = {1: 300e3, 2: 400e3, 3: 500e3}
H0 = H0_MAP[choice]

# ── 初始圆轨道状态 ─────────────────────────────────────────────────────────────
r0 = R_E + H0
v0 = np.sqrt(G * M / r0)
x,  y  = float(r0), 0.0
vx, vy = 0.0, float(v0)
t_sim  = 0.0

# 估算轨道周期（用于轨迹拖尾长度）
T_orb = 2 * np.pi * np.sqrt(r0**3 / (G * M))   # 秒

# ── 动画速度参数 ───────────────────────────────────────────────────────────────
# 每帧推进的仿真时间，默认设为约 1/2 个轨道周期/帧
# 这样每秒(30fps)能看到约 15 个轨道周期（加速 ~450 倍）
STEPS_MIN, STEPS_MAX = 10, 2000
steps_per_frame = [max(STEPS_MIN, int(T_orb / (2 * dt)))]

# 拖尾长度 = 保留约 2 个轨道周期的位置记录
TRAIL_LEN = int(2 * T_orb / dt)
traj_x, traj_y = [], []
finished = [False]

print(f"初始高度: {H0/1e3:.0f} km，圆轨道速度: {v0/1e3:.3f} km/s，轨道周期: {T_orb/60:.1f} min")
print(f"每帧推进 {steps_per_frame[0]} 步 × {dt:.0f} s = {steps_per_frame[0]*dt:.0f} s/帧")

# ── RK4 单步 ──────────────────────────────────────────────────────────────────
def derivatives(state):
    px, py, pvx, pvy = state
    r   = max(np.hypot(px, py), 1e3)
    h   = r - R_E
    ag  = -G * M / r**2
    axg = ag * px / r
    ayg = ag * py / r
    v   = np.hypot(pvx, pvy)
    rho = get_density(h)
    ad  = 0.5 * Cd * rho * A * v**2 / m
    axd = -ad * pvx / v if v > 1e-6 else 0.0
    ayd = -ad * pvy / v if v > 1e-6 else 0.0
    return np.array([pvx, pvy, axg + axd, ayg + ayd])

def rk4(state):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ── 绘图初始化 ─────────────────────────────────────────────────────────────────
BG = "#0d0d1a"
fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
ax.set_facecolor(BG)
ax.set_aspect("equal")

_view = (R_E + H0 * 1.25) / 1e6   # 视图范围（×10³ km）
ax.set_xlim(-_view, _view)
ax.set_ylim(-_view, _view)
ax.tick_params(colors="#555")
for sp in ax.spines.values():
    sp.set_color("#333")

# 地球
earth_patch = plt.Circle((0, 0), R_E / 1e6, color="#1a6b3c", zorder=3)
ax.add_patch(earth_patch)
# 大气层光晕
for alpha, radius in [(0.06, R_E + 80e3), (0.04, R_E + 200e3)]:
    ax.add_patch(plt.Circle((0, 0), radius / 1e6,
                             color="#29b6f6", alpha=alpha, zorder=2))

# 动态元素
trail_line, = ax.plot([], [], color="#ff9800", lw=0.9, alpha=0.7, zorder=4)
sat_dot     = ax.scatter([], [], c="#ffeb3b", s=60,
                          edgecolors="white", linewidths=0.5, zorder=5)

# 文字信息面板
info_box = dict(boxstyle="round,pad=0.4", facecolor="#000000aa",
                edgecolor="#444", linewidth=0.8)
info_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                   color="#e0e0e0", fontsize=9, va="top", ha="left",
                   fontfamily="monospace", bbox=info_box, zorder=6)
speed_txt = ax.text(0.98, 0.03, "", transform=ax.transAxes,
                    color="#888", fontsize=8, va="bottom", ha="right", zorder=6)

ax.set_title(f"人造卫星轨道衰减  初始高度 {H0/1e3:.0f} km",
             color="white", fontsize=11, pad=10)
ax.text(0.5, 0.01,
        "空格：暂停/继续    +/- ：加速/减速    关闭窗口退出",
        transform=ax.transAxes, color="#666", fontsize=8,
        ha="center", va="bottom")

# ── 动画更新函数 ───────────────────────────────────────────────────────────────
_paused = [False]

def update(_):
    global x, y, vx, vy, t_sim
    if _paused[0] or finished[0]:
        return trail_line, sat_dot

    state = np.array([x, y, vx, vy])
    for _ in range(steps_per_frame[0]):
        r = np.hypot(state[0], state[1])
        if r - R_E < H_STOP:
            finished[0] = True
            break
        state = rk4(state)
        t_sim += dt
        traj_x.append(state[0])
        traj_y.append(state[1])

    x, y, vx, vy = state

    # 裁剪拖尾长度
    if len(traj_x) > TRAIL_LEN:
        del traj_x[:-TRAIL_LEN]
        del traj_y[:-TRAIL_LEN]

    # 更新轨迹和卫星点（转换为 ×10³ km）
    tx = np.array(traj_x) / 1e6
    ty = np.array(traj_y) / 1e6
    trail_line.set_data(tx, ty)
    sat_dot.set_offsets([[x / 1e6, y / 1e6]])

    # 当前状态信息
    r_cur = np.hypot(x, y)
    h_cur = (r_cur - R_E) / 1e3
    v_cur = np.hypot(vx, vy) / 1e3
    rho_cur = get_density(r_cur - R_E)
    days  = t_sim / 86400

    if finished[0]:
        info_str = (f"高度:  {h_cur:6.1f} km\n"
                    f"速度:  {v_cur:.3f} km/s\n"
                    f"时间:  {days:.2f} 天\n"
                    f"ρ:  {rho_cur:.2e} kg/m³\n\n"
                    f"★ 已进入大气层 ★")
        info_txt.set_color("#ff5252")
        # 再入闪光
        ax.add_patch(plt.Circle((x/1e6, y/1e6), 0.04,
                                 color="#ff5252", alpha=0.6, zorder=6))
        ani.event_source.stop()
        # 保存截图
        fig.savefig(r"F:\Code\天体运动八年级研学\result.png",
                    dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"模拟结束：{days:.2f} 天后再入，截图已保存 result.png")
    else:
        info_str = (f"高度:  {h_cur:6.1f} km\n"
                    f"速度:  {v_cur:.3f} km/s\n"
                    f"时间:  {days:.2f} 天\n"
                    f"ρ:  {rho_cur:.2e} kg/m³")

    info_txt.set_text(info_str)
    speed_txt.set_text(
        f"速度倍率 ×{steps_per_frame[0] * dt:.0f}s/帧  "
        f"({'暂停' if _paused[0] else '运行中'})"
    )

    return trail_line, sat_dot

# ── 键盘控制 ──────────────────────────────────────────────────────────────────
def on_key(event):
    if event.key == " ":
        _paused[0] = not _paused[0]
    elif event.key in ("+", "="):
        steps_per_frame[0] = min(STEPS_MAX, int(steps_per_frame[0] * 1.5))
        print(f"加速 → {steps_per_frame[0]} 步/帧")
    elif event.key == "-":
        steps_per_frame[0] = max(STEPS_MIN, int(steps_per_frame[0] / 1.5))
        print(f"减速 → {steps_per_frame[0]} 步/帧")

fig.canvas.mpl_connect("key_press_event", on_key)

# ── 启动动画 ──────────────────────────────────────────────────────────────────
ani = FuncAnimation(
    fig, update,
    interval=33,      # ~30 fps
    blit=True,
    repeat=False,
    cache_frame_data=False,
)

plt.tight_layout()
plt.show()
