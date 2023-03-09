import numpy as np
import OSS

path1 = "2Dclasses/aSyn_Benedikt_controll/aSyn_L1A.mrcs"
path2 = "2Dclasses/aSyn_Benedikt_controll/aSyn_L1B.mrcs"
path3 = "2Dclasses/aSyn_Benedikt_controll/aSyn_L1C.mrcs"
path4 = "2Dclasses/aSyn_Benedikt_controll/aSyn_L2A.mrcs"
path5 = "2Dclasses/aSyn_Benedikt_controll/aSyn_L2B_L3A.mrcs"

data1 = OSS.open_mrc(path1)
data2 = OSS.open_mrc(path2)
data3 = OSS.open_mrc(path3)
data4 = OSS.open_mrc(path4)
data5 = OSS.open_mrc(path5)

shape1 = np.shape(data1)
shape2 = np.shape(data2)
shape3 = np.shape(data3)
shape4 = np.shape(data4)
shape5 = np.shape(data5)

print(f"length class 1:  {shape1[0]}")
print(f"length class 2:  {shape2[0]}")
print(f"length class 3:  {shape3[0]}")
print(f"length class 4:  {shape4[0]}")
print(f"length class 5:  {shape5[0]}")

final_length = shape1[0] + shape2[0] + shape3[0] + shape4[0] + shape5[0]
new_arr = np.zeros((final_length, shape1[1], shape1[2]),dtype=np.float32)
new_arr[:shape1[0],:,:] = data1
new_arr[shape1[0]:shape1[0]+shape2[0],:,:] = data2
new_arr[shape1[0]+shape2[0]:shape1[0]+shape2[0]+shape3[0],:,:] = data3
new_arr[shape1[0]+shape2[0]+shape3[0]:shape1[0]+shape2[0]+shape3[0]+shape4[0],:,:] = data4
new_arr[shape1[0]+shape2[0]+shape3[0]+shape4[0]:,:,:] = data5

new_arr.astype(int)

OSS.save_mrc("2Dclasses/aSyn_Benedikt_controll/comb", new_arr)
print("saved succesfully")
