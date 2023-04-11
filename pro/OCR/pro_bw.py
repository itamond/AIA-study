import numpy as np

x=np.random.randint(0,255)
y=np.random.randint(0,255)
z=np.random.randint(0,255)

a = [x, y, z]
print(a)

for i in range(len(a)):
    if a[i]>200:
        a[i]=255
    elif a[i]<200:
        a[i]=0

print(a)
