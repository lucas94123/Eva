
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

lockin=[]
cap1=[]
t=[]
i = 0
filename = '20251121(3) -y 1000 per move(-1V).txt'

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
y1 = savgol_filter(y, 10, 3)

dx = x[2:] - x[:-2]
dy = y1[2:] - y1[:-2]

mask = dx != 0  

x_mid = x[1:-1] 

slopes = dy[mask] / dx[mask]

x_mid = x_mid[mask]

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
plt.plot(x_mid, slopes, marker='o', linestyle='-')
plt.xlabel("Index (between two adjacent points)")
plt.ylabel("Slope (dy/dx)")
plt.title("Adjacent-point Slopes")
plt.grid(True)
plt.savefig(f"{name_without_ext}_slope(no noise)_vs_cap.png")

plt.show()
