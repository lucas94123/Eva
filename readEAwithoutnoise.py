
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,medfilt
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor

lockin=[]
cap1=[]
step = []   
t=[]
i = 0
filename = '20260108_115724_lock-in log.txt'

with open(filename, 'r') as file:
    for line in file:
        parts = line.split(",")
        lock_in_value = 0.0
        cap1_value = 0.0
        step_value = 0.0

        for item in parts:
            

            if"Lock-in:" in item:
                lock_in_value = item.split(":")[1].strip()
            if ("Capacitance 1:1" in item) or ("Capacitance 1:0" in item):
              cap1_value = item.split(":")[1].strip()
            if "Step count:" in item:
                step_value = item.split(":")[1].strip()
                print(step_value)

                           
            
        t.append(i)
        i = i+20
        cap1.append(float(cap1_value))
        lockin.append(float(lock_in_value))
        step.append(int(step_value))
                
            


base = os.path.basename(filename)
name_without_ext = os.path.splitext(base)[0]

x = np.array(cap1)
y = np.array(lockin)
print(len(x), len(y), len(step))

z = np.array(step)

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

plt.figure()
plt.plot(t, lockin, marker='o', linestyle='-')
plt.xlabel('time Step')
plt.ylabel('Lock-in')
plt.title('Lock-in vs time')
plt.grid(True)
plt.savefig(f"{name_without_ext}lockin_vs_time.png")




plt.figure()
plt.plot(t, cap1, marker='o', linestyle='-')
plt.xlabel('Time Step')
plt.ylabel('Capacitance 1')
plt.title('Capacitance 1 over Time')       
plt.grid(True)
plt.savefig(f"{name_without_ext}_cap1_vs_time.png")

plt.figure()
plt.plot(cap1,lockin , marker='o', linestyle='-')
plt.xlabel('cap')
plt.ylabel('lockin')
plt.title('lockin over cap')       
plt.grid(True)
plt.savefig(f"{name_without_ext}_lockin_vs_cap.png")

plt.figure()
plt.plot(z_mid_m, slopes1, marker='o', linestyle='-')
plt.xlabel("Index (between two adjacent points)")
plt.ylabel("Slope (dx/dz)")
plt.title("Adjacent-point Slopes")
plt.grid(True)
plt.savefig(f"{name_without_ext}_slope(no noise)_vs_cap.png")


plt.figure()
plt.plot(z_mid_m, slopes2, marker='o', linestyle='-')
plt.xlabel("Index (between two adjacent points)")
plt.ylabel("Slope (dy/dz)")
plt.title("Adjacent-point Slopes")
plt.grid(True)
plt.savefig(f"{name_without_ext}_slope(no noise)_vs_cap.png")

plt.show()

#=======================
"""
slopes_L = medfilt(slopes, kernel_size=5)
x_mid_L= x_mid.reshape(-1, 1)
lr = LinearRegression().fit(x_mid_L, slopes_L)
y_fit = lr.predict(x_mid_L)

print("y = a*x + b")
print("a =", lr.coef_[0], "b =", lr.intercept_)

# 3) 畫圖
plt.figure()
plt.scatter(x, y, s=15, label="raw")
plt.plot(x_mid_L, slopes_L, label="median filtered")
plt.plot(x_mid_L, y_fit, linewidth=2, label="linear fit")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
"""
#======================\



dz_dx = savgol_filter(
    y,
    window_length=55,   
    polyorder=3,
    deriv=1,
    delta=20
)
plt.figure()
plt.plot(z, dz_dx, marker='o', linestyle='-')
plt.xlabel("Capacitance 1")
plt.ylabel("Smoothed Derivative (dy/dx)")
plt.title("Smoothed Derivative using Savitzky-Golay Filter")
plt.grid(True)
plt.savefig(f"{name_without_ext}_smoothed_derivative_vs_cap.png")
plt.show()
