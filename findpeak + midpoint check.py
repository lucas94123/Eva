import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt, find_peaks
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor

lockin = []
cap1 = []
step = []
t = []
i = 0

filename = "D:\\NTUMSE\\IOPAS\\Eva Au tower\\20260204_110544\\20260204_110544_lock-in log.txt"

with open(filename, 'r') as file:
    for line in file:
        parts = line.split(",")
        lock_in_value = 0.0
        cap1_value = 0.0
        step_value = 0.0

        for item in parts:
            if "Lock-in:" in item:
                lock_in_value = item.split(":")[1].strip()
            if ("Capacitance 1:1" in item) or ("Capacitance 1:0" in item):
                cap1_value = item.split(":")[1].strip()
            if "Step count:" in item:
                step_value = item.split(":")[1].strip()

        t.append(i)
        i = i + 20
        cap1.append(float(cap1_value))
        lockin.append(float(lock_in_value))
        step.append(int(step_value))

base = os.path.basename(filename)
name_without_ext = os.path.splitext(base)[0]

x = np.array(cap1)
y = np.array(lockin)
z = np.array(step)

print("len(x), len(y), len(z) =", len(x), len(y), len(z))

# =======================
# Your existing smoothing and slope calculations
# =======================
y1 = savgol_filter(y, 55, 3)

dz = z[2:] - z[:-2]
dx = x[2:] - x[:-2]
dy = y1[2:] - y1[:-2]

mask = dz != 0
z_mid = z[1:-1]
x_mid = x[1:-1]

slopes1 = dx[mask] / dz[mask]
slopes2 = dy[mask] / dz[mask]

z_mid_m = z_mid[mask]
x_mid_m = x_mid[mask]

# =======================
# Plots you already had
# =======================
plt.figure()
plt.plot(t, lockin, marker='o', linestyle='-')
plt.xlabel('time Step')
plt.ylabel('Lock-in')
plt.title('Lock-in vs time')
plt.grid(True)
plt.savefig(f"{name_without_ext}_lockin_vs_time.png")

plt.figure()
plt.plot(t, cap1, marker='o', linestyle='-')
plt.xlabel('Time Step')
plt.ylabel('Capacitance 1')
plt.title('Capacitance 1 over Time')
plt.grid(True)
plt.savefig(f"{name_without_ext}_cap1_vs_time.png")

plt.figure()
plt.plot(cap1, lockin, marker='o', linestyle='-')
plt.xlabel('Capacitance 1')
plt.ylabel('Lock-in')
plt.title('Lock-in vs Capacitance 1')
plt.grid(True)
plt.savefig(f"{name_without_ext}_lockin_vs_cap.png")

plt.figure()
plt.plot(z_mid_m, slopes1, marker='o', linestyle='-')
plt.xlabel("Step (z_mid)")
plt.ylabel("Slope (dx/dz)")
plt.title("Adjacent-point Slopes (dx/dz)")
plt.grid(True)
plt.savefig(f"{name_without_ext}_slope1_vs_step.png")

plt.figure()
plt.plot(z_mid_m, slopes2, marker='o', linestyle='-')
plt.xlabel("Step (z_mid)")
plt.ylabel("Slope (dy/dz)")
plt.title("Adjacent-point Slopes (dy/dz)")
plt.grid(True)
plt.savefig(f"{name_without_ext}_slope2_vs_step.png")

plt.show()

# =======================
# Smoothed derivative + peak & valley detection
# =======================
dz_dx = savgol_filter(
    y,
    window_length=55,
    polyorder=3,
    deriv=1,
    delta=20
)

# ---- Peak/valley parameters (tune if needed) ----
prom = 1.0 * np.std(dz_dx)   # prominence threshold
dist = 20                    # minimum distance between peaks
# -------------------------------------------------

# Peaks (positive)
peaks, pprops = find_peaks(dz_dx, prominence=prom, distance=dist)

# Valleys (negative): find peaks in -dz_dx
valleys, vprops = find_peaks(-dz_dx, prominence=prom, distance=dist)

# =======================
# Print coordinates (z, dz_dx) and also (cap1, dz_dx)
# =======================
print("\n==== Peaks coordinates ====")
for idx in peaks:
    print(f"peak: index={idx}, z={z[idx]}, cap1={x[idx]:.6f}, dz_dx={dz_dx[idx]:.6e}")

print("\n==== Valleys coordinates ====")
for idx in valleys:
    print(f"valley: index={idx}, z={z[idx]}, cap1={x[idx]:.6f}, dz_dx={dz_dx[idx]:.6e}")

# (Optional) save to txt
np.savetxt(
    f"{name_without_ext}_peaks_z_cap_dzdx.txt",
    np.column_stack([peaks, z[peaks], x[peaks], dz_dx[peaks]]),
    header="index z cap1 dz_dx",
    fmt=["%d", "%d", "%.8f", "%.8e"]
)
np.savetxt(
    f"{name_without_ext}_valleys_z_cap_dzdx.txt",
    np.column_stack([valleys, z[valleys], x[valleys], dz_dx[valleys]]),
    header="index z cap1 dz_dx",
    fmt=["%d", "%d", "%.8f", "%.8e"]
)

# =======================
# Plot derivative vs step count (z) with peaks/valleys marked
# =======================
plt.figure()
plt.plot(z, dz_dx, marker='o', linestyle='-', label='dz_dx')
plt.scatter(z[peaks], dz_dx[peaks], s=120, marker='x', label='peaks')
plt.scatter(z[valleys], dz_dx[valleys], s=120, marker='x', label='valleys')

for idx in peaks:
    plt.annotate(f"P\n({z[idx]}, {dz_dx[idx]:.2e})",
                 (z[idx], dz_dx[idx]),
                 textcoords="offset points", xytext=(6, 8), ha='left')

for idx in valleys:
    plt.annotate(f"V\n({z[idx]}, {dz_dx[idx]:.2e})",
                 (z[idx], dz_dx[idx]),
                 textcoords="offset points", xytext=(6, -18), ha='left')

plt.xlabel("Step count (z)")
plt.ylabel("Smoothed Derivative (dy/dx)")
plt.title("Smoothed Derivative (Savitzky-Golay) + Peaks/Valleys")
plt.grid(True)
plt.legend()
plt.savefig(f"{name_without_ext}_smoothed_derivative_vs_step_peaks_valleys.png")
plt.show()

# =======================
# (Optional) Plot derivative vs capacitance (cap1) with peaks/valleys marked
# =======================
plt.figure()
plt.plot(x, dz_dx, marker='o', linestyle='-', label='dz_dx')
plt.scatter(x[peaks], dz_dx[peaks], s=120, marker='x', label='peaks')
plt.scatter(x[valleys], dz_dx[valleys], s=120, marker='x', label='valleys')

for idx in peaks:
    plt.annotate(f"P\n({x[idx]:.0f}, {dz_dx[idx]:.2e})",
                 (x[idx], dz_dx[idx]),
                 textcoords="offset points", xytext=(6, 8), ha='left')

for idx in valleys:
    plt.annotate(f"V\n({x[idx]:.0f}, {dz_dx[idx]:.2e})",
                 (x[idx], dz_dx[idx]),
                 textcoords="offset points", xytext=(6, -18), ha='left')

plt.xlabel("Capacitance 1 (cap1)")
plt.ylabel("Smoothed Derivative (dy/dx)")
plt.title("Smoothed Derivative vs Capacitance 1 + Peaks/Valleys")
plt.grid(True)
plt.legend()
plt.savefig(f"{name_without_ext}_smoothed_derivative_vs_cap1_peaks_valleys.png")

plt.show()

# =======================
# 點對稱檢測：相鄰 Peak-Valley 中點 + 振幅對稱
# =======================

# ---- 可調參數 ----
midpoint_tol_ratio = 0.2   # 中點 y 值容許誤差（相對於該區段 dz_dx 標準差的倍數）
amplitude_tol_ratio = 0.3  # 振幅容許誤差（相對於兩者平均振幅的比例）
# ------------------

# 將 peaks 和 valleys 合併，並標記類型，依 z 排序
all_extrema = []
for idx in peaks:
    all_extrema.append((z[idx], idx, 'peak'))
for idx in valleys:
    all_extrema.append((z[idx], idx, 'valley'))
all_extrema.sort(key=lambda item: item[0])

# 找相鄰異類配對（peak-valley 或 valley-peak）
pairs = []
for i in range(len(all_extrema) - 1):
    z1, idx1, type1 = all_extrema[i]
    z2, idx2, type2 = all_extrema[i + 1]
    if type1 != type2:
        pairs.append((idx1, type1, idx2, type2))

print("\n==== 點對稱檢測結果 ====")
print(f"{'配對':<6} {'類型':<15} {'中點z':<10} {'中點dzdx(理論)':<20} {'中點dzdx(實際)':<20} {'中點通過':<10} {'振幅差%':<10} {'振幅通過':<10}")

sym_results = []

for pair_num, (idx1, type1, idx2, type2) in enumerate(pairs):
    z1, y1_val = z[idx1], dz_dx[idx1]
    z2, y2_val = z[idx2], dz_dx[idx2]

    # 取出這對 peak-valley 之間的區段，計算局部標準差
    seg_start = min(idx1, idx2)
    seg_end   = max(idx1, idx2)
    segment   = dz_dx[seg_start:seg_end + 1]
    midpoint_tol = midpoint_tol_ratio * np.std(segment)

    # 中點座標
    z_mid_val    = (z1 + z2) / 2.0
    y_mid_theory = (y1_val + y2_val) / 2.0  # 理論中點 y（兩端平均）

    # 內插取得圖形上 z_mid 對應的實際 dz_dx 值
    y_mid_actual = np.interp(z_mid_val, z, dz_dx)

    # 判斷中點是否在圖形上
    midpoint_ok = abs(y_mid_actual - y_mid_theory) <= midpoint_tol

    # 振幅對稱：以實際中點 y 值為基線（而非 0），分別計算 peak 與 valley 相對基線的距離
    baseline       = y_mid_actual
    amp1           = abs(y1_val - baseline)
    amp2           = abs(y2_val - baseline)
    amp_mean       = (amp1 + amp2) / 2.0
    amp_diff_ratio = abs(amp1 - amp2) / amp_mean * 100  # 百分比
    amplitude_ok   = abs(amp1 - amp2) <= amplitude_tol_ratio * amp_mean

    # 整體判斷
    is_symmetric = midpoint_ok and amplitude_ok

    sym_results.append({
        'pair': pair_num + 1,
        'idx1': idx1, 'type1': type1, 'idx2': idx2, 'type2': type2,
        'z_mid': z_mid_val, 'y_mid_theory': y_mid_theory, 'y_mid_actual': y_mid_actual,
        'midpoint_ok': midpoint_ok, 'amp_diff_ratio': amp_diff_ratio,
        'amplitude_ok': amplitude_ok, 'is_symmetric': is_symmetric
    })

    print(f"#{pair_num+1:<5} {type1+'-'+type2:<15} {z_mid_val:<10.1f} "
          f"{y_mid_theory:<20.4e} {y_mid_actual:<20.4e} "
          f"{'O' if midpoint_ok else 'X':<10} "
          f"{amp_diff_ratio:<10.1f} "
          f"{'O' if amplitude_ok else 'X':<10}  "
          f"=> {'[Symmetric]' if is_symmetric else '[Asymmetric]'}")

# =======================
# 畫圖：在 dz_dx vs z 圖上標記中點與對稱結果
# =======================
plt.figure(figsize=(12, 6))
plt.plot(z, dz_dx, linestyle='-', color='steelblue', label='dz_dx')
plt.scatter(z[peaks],   dz_dx[peaks],   s=120, marker='^', color='red',  zorder=5, label='Peaks')
plt.scatter(z[valleys], dz_dx[valleys], s=120, marker='v', color='blue', zorder=5, label='Valleys')

# 分開收集對稱與不對稱的中點，最後各畫一次以產生正確圖例
sym_zs,  sym_ys  = [], []
asym_zs, asym_ys = [], []

for r in sym_results:
    color = 'green' if r['is_symmetric'] else 'orange'

    # 畫連線（peak 到 valley）
    idx1, idx2 = r['idx1'], r['idx2']
    plt.plot([z[idx1], z[idx2]], [dz_dx[idx1], dz_dx[idx2]],
             linestyle='--', color=color, alpha=0.5, linewidth=1)

    # 標註文字（使用 ASCII 避免字型問題）
    plt.annotate(f"#{r['pair']} {'[sym]' if r['is_symmetric'] else '[asym]'}",
                 (r['z_mid'], r['y_mid_actual']),
                 textcoords="offset points", xytext=(0, 12),
                 ha='center', fontsize=8, color=color)

    if r['is_symmetric']:
        sym_zs.append(r['z_mid'])
        sym_ys.append(r['y_mid_actual'])
    else:
        asym_zs.append(r['z_mid'])
        asym_ys.append(r['y_mid_actual'])

# 各畫一次，label 使用純 ASCII 確保圖例正常顯示
if sym_zs:
    plt.scatter(sym_zs, sym_ys, s=180, marker='*', color='green',
                zorder=6, label='Symmetric midpoint')
if asym_zs:
    plt.scatter(asym_zs, asym_ys, s=150, marker='X', color='orange',
                zorder=6, label='Asymmetric midpoint')

plt.xlabel("Step count (z)")
plt.ylabel("Smoothed Derivative (dy/dx)")
plt.title("Point Symmetry Check: Peak-Valley Midpoint + Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()