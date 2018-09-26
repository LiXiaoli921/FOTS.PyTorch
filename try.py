# lis = [0, 1]
# a = ','.join([str(i) for i in lis])
# print(a)
import numpy as np



def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


a1 = np.cross([130,2],[-11,-11])
print(a1)
a2 = np.linalg.norm(a1)
print(a2,'a2 is '*10)# 1408.0
q3 = np.linalg.norm([130,2])
print(q3,'q3 is '*10) #130.01538370516005

a4 = a2/q3
print('a4 is ',a4)# a4 is  10.829487710415604




h,w = 512,512

geo_map = np.zeros((h, w, 5), dtype = np.float32)

p0_rect = np.array([305,35])
p1_rect = np.array([435,37])
point = np.array([316,46])
y = []
x = []
geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
print(geo_map)