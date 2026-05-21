"""
人造卫星轨道衰减模拟
基于 NRLMSISE-00 标准大气数据表 + RK4 数值积分
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── 常数 ──────────────────────────────────────────────────────────────────────
G   = 6.674e-11       # 万有引力常数 N·m²/kg²
M   = 5.972e24        # 地球质量 kg
R_E = 6371e3          # 地球半径 m
Cd  = 2.2             # 阻力系数
A   = 10.0            # 卫星截面积 m²
m   = 500.0           # 卫星质量 kg
dt  = 10.0            # 时间步长 s
H0  = 400e3           # 初始高度 m（国际空间站）
H_STOP = 100e3        # 停止高度 m

# ── NRLMSISE-00 标准大气密度数据表（中等太阳活动，赤道，日侧）─────────────────
# 数据来源：NRLMSISE-00 模型，F10.7=150, Ap=4
# 高度 km : 质量密度 kg/m³
_NRLMSISE_TABLE = np.array([
    [  0,   1.225e+00],
    [ 10,   4.135e-01],
    [ 20,   8.891e-02],
    [ 30,   1.841e-02],
    [ 40,   3.996e-03],
    [ 50,   1.027e-03],
    [ 60,   3.097e-04],
    [ 70,   8.283e-05],
    [ 80,   1.846e-05],
    [ 90,   3.416e-06],
    [100,   5.297e-07],
    [110,   9.648e-08],
    [120,   2.438e-08],
    [130,   8.484e-09],
    [140,   3.845e-09],
    [150,   2.070e-09],
    [160,   1.233e-09],
    [180,   5.194e-10],
    [200,   2.541e-10],
    [220,   1.367e-10],
    [250,   6.073e-11],
    [300,   1.916e-11],
    [350,   7.014e-12],
    [400,   2.803e-12],
    [450,   1.184e-12],
    [500,   5.215e-13],
    [550,   2.384e-13],
    [600,   1.137e-13],
    [700,   2.818e-14],
    [800,   7.998e-15],
    [900,   2.490e-15],
    [1000,  8.510e-16],
])

_h_table   = _NRLMSISE_TABLE[:, 0]   # km
_rho_table = _NRLMSISE_TABLE[:, 1]   # kg/m³
_log_rho   = np.log(_rho_table)       # 对数插值更精准


def get_density(h_m: float) -> float:
    """NRLMSISE-00 表格插值，输入 m，返回 kg/m³"""
    h_km = h_m / 1e3
    h_km = float(np.clip(h_km, _h_table[0], _h_table[-1]))
    log_rho = np.interp(h_km, _h_table, _log_rho)
    return float(np.exp(log_rho))


# ── 运动方程 ──────────────────────────────────────────────────────────────────

def equations(state: np.ndarray) -> np.ndarray:
    """state = [x, y, vx, vy]，返回 [vx, vy, ax, ay]"""
    x, y, vx, vy = state
    r  = np.hypot(x, y)
    h  = r - R_E

    # 重力
    a_g  = -G * M / r**2
    ax_g = a_g * x / r
    ay_g = a_g * y / r

    # 阻力：F = 0.5 × Cd × ρ × A × v²
    v   = np.hypot(vx, vy)
    rho = get_density(h)
    a_d = 0.5 * Cd * rho * A * v**2 / m
    ax_d = -a_d * vx / v if v > 1e-6 else 0.0
    ay_d = -a_d * vy / v if v > 1e-6 else 0.0

    return np.array([vx, vy, ax_g + ax_d, ay_g + ay_d])


def rk4_step(state: np.ndarray, dt_: float) -> np.ndarray:
    k1 = equations(state)
    k2 = equations(state + 0.5 * dt_ * k1)
    k3 = equations(state + 0.5 * dt_ * k2)
    k4 = equations(state + dt_ * k3)
    return state + (dt_ / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ── 初始圆轨道 ────────────────────────────────────────────────────────────────
r0 = R_E + H0
v0 = np.sqrt(G * M / r0)
state = np.array([r0, 0.0, 0.0, v0])

print(f"初始高度: {H0/1e3:.0f} km，圆轨道速度: {v0/1e3:.3f} km/s")
print("开始 RK4 模拟（步长 10 s）…")

# ── 积分 ──────────────────────────────────────────────────────────────────────
# 只保存每 60 步（10 min）一帧，避免内存溢出
SAVE_INTERVAL = 60

xs, ys, heights, times, densities = [], [], [], [], []
t = 0.0

for step in range(20_000_000):
    x, y, vx, vy = state
    r = np.hypot(x, y)
    h = r - R_E

    if h < H_STOP:
        print(f"高度降至 {h/1e3:.1f} km（< 100 km），停止。")
        break

    if step % SAVE_INTERVAL == 0:
        xs.append(x);  ys.append(y)
        heights.append(h / 1e3)
        times.append(t / 86400.0)
        densities.append(get_density(h))

    state = rk4_step(state, dt)
    t    += dt

    if step % 1_000_000 == 0 and step > 0:
        print(f"  {t/86400:.1f} 天，高度 {h/1e3:.1f} km")

total_days = t / 86400.0
print(f"模拟完成：{total_days:.2f} 天，{len(xs)} 个记录点")

xs        = np.array(xs) / 1e6    # → ×10³ km，量级合适
ys        = np.array(ys) / 1e6
heights   = np.array(heights)
times     = np.array(times)
densities = np.array(densities)

# ── 绘图 ──────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BG = '#0a0a1a'
fig = plt.figure(figsize=(18, 7), facecolor=BG, dpi=150)
fig.suptitle('人造卫星轨道衰减模拟  (基于 NRLMSISE-00 大气密度模型)',
             color='white', fontsize=14)

# 手动设置子图位置，避免 tight_layout 与 equal aspect 冲突
ax1 = fig.add_axes([0.04, 0.10, 0.28, 0.80], facecolor=BG)
ax2 = fig.add_axes([0.40, 0.12, 0.25, 0.78], facecolor=BG)
ax3 = fig.add_axes([0.72, 0.12, 0.25, 0.78], facecolor=BG)

norm_t = (times - times[0]) / max(times[-1] - times[0], 1e-9)

# ─ 子图 1：轨道 x-y ──────────────────────────────────────────────────────────
ax1.set_aspect('equal', adjustable='box')
ax1.set_facecolor(BG)

earth = Circle((0, 0), R_E / 1e6, color='#1a6b3c', zorder=3, label='地球')
ax1.add_patch(earth)
atm   = Circle((0, 0), (R_E + 150e3) / 1e6, color='#29b6f6', alpha=0.12, zorder=2)
ax1.add_patch(atm)

step_plot = max(1, len(xs) // 3000)
for i in range(0, len(xs) - step_plot, step_plot):
    c = plt.cm.plasma(norm_t[i])
    ax1.plot(xs[i:i+step_plot+1], ys[i:i+step_plot+1], color=c, lw=0.5, alpha=0.8)

ax1.scatter(xs[0],  ys[0],  color='#69f0ae', s=30, zorder=5, label='起点')
ax1.scatter(xs[-1], ys[-1], color='#ff5252', s=30, zorder=5, label='终点')

_r = R_E / 1e6 * 1.15
ax1.set_xlim(-_r, _r)
ax1.set_ylim(-_r, _r)
ax1.set_xlabel('x  (×10³ km)', color='#ccc', fontsize=9)
ax1.set_ylabel('y  (×10³ km)', color='#ccc', fontsize=9)
ax1.set_title('卫星轨道（颜色 = 时间进程）', color='white', fontsize=10, pad=6)
ax1.tick_params(colors='#aaa', labelsize=7)
for sp in ax1.spines.values(): sp.set_color('#333')
ax1.legend(fontsize=7, facecolor='#111', labelcolor='white',
           loc='lower right', framealpha=0.6)

# colorbar 嵌入 ax1 右侧
cax = fig.add_axes([0.325, 0.12, 0.012, 0.78])
sm  = plt.cm.ScalarMappable(cmap='plasma',
      norm=plt.Normalize(vmin=times[0], vmax=times[-1]))
sm.set_array([])
cb  = fig.colorbar(sm, cax=cax)
cb.set_label('时间 (天)', color='white', fontsize=8)
cb.ax.yaxis.set_tick_params(color='white', labelsize=7)
plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

# ─ 子图 2：高度-时间 ──────────────────────────────────────────────────────────
ax2.plot(times, heights, color='#4fc3f7', lw=1.5)
ax2.axhline(100, color='#ef5350', lw=1.2, ls='--', alpha=0.9)
ax2.axhline(400, color='#66bb6a', lw=1.2, ls='--', alpha=0.9)
ax2.text(times[-1]*0.02, 103, '100 km 再入线', color='#ef5350', fontsize=7, va='bottom')
ax2.text(times[-1]*0.02, 403, '400 km ISS 初始', color='#66bb6a', fontsize=7, va='bottom')

# 终点标注
ax2.annotate(f'{times[-1]:.1f} 天\n{heights[-1]:.0f} km',
             xy=(times[-1], heights[-1]),
             xytext=(times[-1]*0.65, heights[-1] + 50),
             color='#ef5350', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#ef5350', lw=1.2))

ax2.set_xlabel('时间 (天)', color='#ccc', fontsize=9)
ax2.set_ylabel('轨道高度 (km)', color='#ccc', fontsize=9)
ax2.set_title('轨道高度随时间变化', color='white', fontsize=10, pad=6)
ax2.tick_params(colors='#aaa', labelsize=8)
ax2.set_ylim(80, 450)
ax2.set_xlim(left=0)
for sp in ax2.spines.values(): sp.set_color('#333')
ax2.grid(alpha=0.15, color='#555')

# ─ 子图 3：大气密度-高度 ──────────────────────────────────────────────────────
h_ref   = np.linspace(100, 500, 300)
rho_ref = np.array([get_density(h * 1e3) for h in h_ref])
ax3.semilogy(rho_ref, h_ref, color='#ffb74d', lw=2)

rho_start = get_density(H0)
rho_end   = get_density(heights[-1] * 1e3)
ax3.scatter(rho_start, H0 / 1e3,    color='#69f0ae', s=60, zorder=5,
            label=f'起点  {H0/1e3:.0f} km')
ax3.scatter(rho_end, heights[-1],   color='#ff5252', s=60, zorder=5,
            label=f'终点  {heights[-1]:.0f} km')

ratio = rho_end / rho_start
ax3.text(rho_start * 4, (H0/1e3 + heights[-1]) / 2,
         f'密度比\n{ratio:.0f}×',
         color='white', fontsize=8, ha='left', va='center')

ax3.set_xlabel('大气密度 (kg/m³)', color='#ccc', fontsize=9)
ax3.set_ylabel('高度 (km)', color='#ccc', fontsize=9)
ax3.set_title('大气密度随高度变化', color='white', fontsize=10, pad=6)
ax3.tick_params(colors='#aaa', labelsize=8)
ax3.set_ylim(80, 520)
for sp in ax3.spines.values(): sp.set_color('#333')
ax3.legend(fontsize=8, facecolor='#111', labelcolor='white',
           loc='upper right', framealpha=0.6)
ax3.grid(alpha=0.15, color='#555', which='both')

# ── 底部参数说明 ──────────────────────────────────────────────────────────────
stats = (f"Cd={Cd}   A={A} m²   m={m} kg   dt={dt:.0f} s  |  "
         f"初始高度 {H0/1e3:.0f} km  →  {heights[-1]:.0f} km  |  "
         f"总时长 {total_days:.1f} 天  |  积分步数 {int(total_days*86400/dt):,}")
fig.text(0.5, 0.01, stats, ha='center', va='bottom', color='#777', fontsize=8)

out = r'F:\Code\天体运动八年级研学\result.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"图像已保存：{out}")
plt.close()