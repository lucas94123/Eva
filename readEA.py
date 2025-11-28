import numpy as np
import os
import matplotlib.pyplot as plt

lockin=[]
cap1=[]
t=[]
i = 0
filename = "D:/NTUMSE/IOPAS/Eva Au tower/20251121(2) +y 1000 per move.txt"

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


plt.plot(t, lockin, marker='o', linestyle='-')
plt.xlabel('time Step')
plt.ylabel('Lock-in')
plt.title('Lock-in vs time')
plt.grid(True)
plt.savefig(f"{name_without_ext}lockin_vs_time.png")

plt.show()


plt.plot(t, cap1, marker='o', linestyle='-')
plt.xlabel('Time Step')
plt.ylabel('Capacitance 1')
plt.title('Capacitance 1 over Time')       
plt.grid(True)
plt.savefig(f"{name_without_ext}_cap1_vs_time.png")

plt.show()
