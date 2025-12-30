
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

lockin=[]
cap1=[]
t=[]
i = 0
filename = "20251219_170231_lock-in log.txt"

with open(filename, 'r') as file:
    for line in file:
        parts = line.split(",")
        for item in parts:
            
           
            if"Lock-in:" in item:
                lock_in_value = item.split(":")[1].strip()
            if "Capacitance 1:0" in item:
                cap1_value = item.split(":")[1].strip()
                           
                if cap1_value != "":
                    t.append(i)
                    i += 1
                    cap1.append(float(cap1_value))
                    lockin.append(float(lock_in_value))
                else:
                     # Skip this entry if Capacitance 1 is missing
                     continue


base = os.path.basename(filename)
name_without_ext = os.path.splitext(base)[0]

x = np.array(cap1)
y = np.array(lockin)


dx = x[2:] - x[:-2]
dy = y[2:] - y[:-2]

mask = dx != 0  

x_mid = x[1:-1] 

slopes = dy[mask] / dx[mask]

x_mid = x_mid[mask]



plt.figure()
plt.plot(x_mid, slopes, marker='o', linestyle='-')
plt.xlabel("Index (between two adjacent points)")
plt.ylabel("Slope (dy/dx)")
plt.title("Adjacent-point Slopes")
plt.grid(True)
plt.savefig(f"{name_without_ext}_slope(orignal)_vs_cap.png")

plt.show()
