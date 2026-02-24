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

filename = '20260211_175333_lock-in log (1).txt'

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
