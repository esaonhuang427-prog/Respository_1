"""
生成 result_math.png
三联静态图：轨道 / 高度-时间衰减曲线 / 大气密度-高度曲线
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── NRLMSISE-00 大气密度表 ─────────────────────────────────────────────────────
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
_h_km    = _TABLE[:, 0]
_log_rho = np.log(_TABLE[:, 1])

def get_density(h_m):
    h_km = float(np.clip(h_m / 1e3, _h_km[0], _h_km[-1]))
    return float(np.exp(np.interp(h_km, _h_km, _log_rho)))

# ── 物理常数 ───────────────────────────────────────────────────────────────────
G      = 6.674e-11
M      = 5.972e24
R_E    = 6371e3
Cd     = 2.2
A      = 10.0
m      = 500.0
dt     = 10.0
H0     = 400e3
H_STOP = 100e3

# ── RK4 ───────────────────────────────────────────────────────────────────────
def derivs(s):
    px, py, pvx, pvy = s
    r   = max(np.hypot(px, py), 1e3)
    h   = r - R_E
    ag  = -G * M / r**2
    v   = np.hypot(pvx, pvy)
    rho = get_density(h)
    ad  = 0.5 * Cd * rho * A * v**2 / m
    axd = -ad * pvx / v if v > 1e-6 else 0.0
    ayd = -ad * pvy / v if v > 1e-6 else 0.0
    return np.array([pvx, pvy, ag * px / r + axd, ag * py / r + ayd])

def rk4(s):
    k1 = derivs(s)
    k2 = derivs(s + 0.5 * dt * k1)
    k3 = derivs(s + 0.5 * dt * k2)
    k4 = derivs(s + dt * k3)
    return s + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)

# ── 积分 ──────────────────────────────────────────────────────────────────────
r0    = R_E + H0
state = np.array([r0, 0.0, 0.0, np.sqrt(G * M / r0)])
t     = 0.0

SAVE = 60   # 每 60 步（10 min）存一帧
xs, ys, heights, times, densities, energies = [], [], [], [], [], []

print("模拟中…")
for step in range(20_000_000):
    px, py, pvx, pvy = state
    r = np.hypot(px, py)
    h = r - R_E
    if h < H_STOP:
        print(f"再入：{t/86400:.2f} 天，高度 {h/1e3:.1f} km")
        break
    if step % SAVE == 0:
        v   = np.hypot(pvx, pvy)
        rho = get_density(h)
        E   = 0.5 * m * v**2 - G * M * m / r   # 机械能 J
        xs.append(px);         ys.append(py)
        heights.append(h / 1e3)
        times.append(t / 86400.0)
        densities.append(rho)
        energies.append(E / 1e9)               # → GJ
    state = rk4(state)
    t    += dt

total_days = t / 86400.0
xs  = np.array(xs)  / 1e6
ys  = np.array(ys)  / 1e6
heights   = np.array(heights)
times     = np.array(times)
densities = np.array(densities)
energies  = np.array(energies)
norm_t    = (times - times[0]) / (times[-1] - times[0])

print(f"完成：{total_days:.2f} 天，{len(times)} 个记录点")

# ── 绘图 ──────────────────────────────────────────────────────────────────────
plt.rcParams["font.family"]        = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#0a0a1a"

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(17, 7),
    facecolor=BG,
    gridspec_kw={"width_ratios": [1, 1.1, 1], "wspace": 0.38},
)
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2, ax3):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color("#333")

fig.suptitle(
    "人造卫星轨道衰减  ·  数学关系图\n"
    "(NRLMSISE-00 大气密度模型 / RK4 积分 / dt = 10 s)",
    color="white", fontsize=13, y=1.01,
)

_r = R_E / 1e6 * 1.16

# ─ 子图 1：轨道 x-y ───────────────────────────────────────────────────────────
ax1.set_aspect("equal", adjustable="box")
ax1.add_patch(Circle((0, 0), R_E / 1e6,           color="#1a6b3c", zorder=3))
ax1.add_patch(Circle((0, 0), (R_E + 150e3) / 1e6, color="#29b6f6", alpha=0.10))
# 初始轨道虚线圆
theta = np.linspace(0, 2 * np.pi, 300)
r_init = (R_E + H0) / 1e6
ax1.plot(r_init * np.cos(theta), r_init * np.sin(theta),
         color="white", lw=0.5, alpha=0.15, ls="--")

step_p = max(1, len(xs) // 3000)
for i in range(0, len(xs) - step_p, step_p):
    ax1.plot(xs[i:i+step_p+1], ys[i:i+step_p+1],
             color=plt.cm.plasma(norm_t[i]), lw=0.5, alpha=0.85)

ax1.scatter(xs[0],  ys[0],  color="#69f0ae", s=30, zorder=5, label="起点")
ax1.scatter(xs[-1], ys[-1], color="#ff5252", s=30, zorder=5, label="终点")
ax1.set_xlim(-_r, _r);  ax1.set_ylim(-_r, _r)
ax1.set_xlabel("x  (×10³ km)", color="#ccc", fontsize=9)
ax1.set_ylabel("y  (×10³ km)", color="#ccc", fontsize=9)
ax1.set_title("卫星轨道（颜色 = 时间进程）", color="white", fontsize=10, pad=6)
ax1.tick_params(colors="#aaa", labelsize=7)
ax1.legend(fontsize=7, facecolor="#111", labelcolor="white",
           loc="lower right", framealpha=0.7)

sm = plt.cm.ScalarMappable(cmap="plasma",
     norm=plt.Normalize(vmin=times[0], vmax=times[-1]))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
cb.set_label("时间 (天)", color="white", fontsize=8)
cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

# ─ 子图 2：高度-时间  +  机械能-时间（双 y 轴）─────────────────────────────────
ax2b = ax2.twinx()
ax2b.set_facecolor(BG)
for sp in ax2b.spines.values(): sp.set_color("#333")

ax2.plot(times, heights,  color="#4fc3f7", lw=1.8, label="轨道高度")
ax2b.plot(times, energies, color="#ff8a65", lw=1.2,
          ls="--", alpha=0.85, label="机械能")

ax2.axhline(100, color="#ef5350", lw=1.0, ls=":", alpha=0.8)
ax2.axhline(400, color="#66bb6a", lw=1.0, ls=":", alpha=0.8)
ax2.text(times[-1] * 0.02, 104, "100 km 再入线",  color="#ef5350", fontsize=7)
ax2.text(times[-1] * 0.02, 404, "400 km ISS 初始", color="#66bb6a", fontsize=7)

# 终点标注（坐标确保在轴范围内）
ax2.annotate(
    f"{times[-1]:.1f} 天\n{heights[-1]:.0f} km",
    xy=(times[-1], heights[-1]),
    xytext=(times[-1] * 0.58, heights[-1] + 60),
    color="#ef5350", fontsize=8,
    arrowprops=dict(arrowstyle="->", color="#ef5350", lw=1.1),
)

ax2.set_xlabel("时间 (天)", color="#ccc", fontsize=9)
ax2.set_ylabel("轨道高度 (km)", color="#4fc3f7", fontsize=9)
ax2b.set_ylabel("机械能 (GJ)", color="#ff8a65", fontsize=9)
ax2.set_title("轨道高度 & 机械能随时间变化", color="white", fontsize=10, pad=6)
ax2.tick_params(colors="#aaa", labelsize=8)
ax2b.tick_params(colors="#ff8a65", labelsize=7)
ax2.set_xlim(0, times[-1] * 1.02)
ax2.set_ylim(80, 450)
ax2.grid(alpha=0.12, color="#555")
lines = ax2.get_lines() + ax2b.get_lines()
ax2.legend(lines, [l.get_label() for l in lines],
           fontsize=7, facecolor="#111", labelcolor="white",
           loc="upper right", framealpha=0.7)

# ─ 子图 3：大气密度-高度 ───────────────────────────────────────────────────────
h_ref   = np.linspace(100, 500, 400)
rho_ref = np.array([get_density(h * 1e3) for h in h_ref])
ax3.semilogy(rho_ref, h_ref, color="#ffb74d", lw=2.2, label="NRLMSISE-00")

rho_s = get_density(H0)
rho_e = get_density(heights[-1] * 1e3)
ax3.scatter(rho_s, H0 / 1e3,    color="#69f0ae", s=70, zorder=5,
            label=f"起点  {H0/1e3:.0f} km")
ax3.scatter(rho_e, heights[-1], color="#ff5252", s=70, zorder=5,
            label=f"终点  {heights[-1]:.0f} km")

# 用 transAxes 坐标，彻底避免超出数据范围
ax3.text(0.35, 0.30,
         f"密度比\n{rho_e/rho_s:.0f}×",
         transform=ax3.transAxes,
         color="white", fontsize=9, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff18", edgecolor="#555"))

ax3.set_xlabel("大气密度  ρ (kg/m³)", color="#ccc", fontsize=9)
ax3.set_ylabel("高度  h (km)", color="#ccc", fontsize=9)
ax3.set_title("大气密度随高度变化  ρ(h)", color="white", fontsize=10, pad=6)
ax3.tick_params(colors="#aaa", labelsize=8)
ax3.set_ylim(80, 520)
ax3.legend(fontsize=8, facecolor="#111", labelcolor="white",
           loc="upper right", framealpha=0.7)
ax3.grid(alpha=0.12, color="#555", which="both")

# ── 底部参数说明 ──────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.01,
    f"Cd = {Cd}   A = {A} m²   m = {m} kg   dt = {dt:.0f} s  │  "
    f"初始高度 {H0/1e3:.0f} km → 再入 {heights[-1]:.0f} km  │  "
    f"总时长 {total_days:.1f} 天  │  积分步数 {int(total_days*86400/dt):,}",
    ha="center", va="top", color="#666", fontsize=7.5,
)

# ── 保存 ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
out = r"F:\Code\天体运动八年级研学\result_math.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"已保存：{out}")
plt.close()
