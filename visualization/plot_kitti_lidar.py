
import numpy as np
lidar = np.fromfile('/lhome/aasmunhb/kitti_data/000007.bin', dtype=np.float32)
lidar = lidar.reshape((-1, 4))
print("Shape of lidar: ", lidar.shape)
x = lidar[:, 0]
y = lidar[:, 1]
z = lidar[:, 2]
print("Min/max x ;", x.min(), x.max())
print("Min/max y ;", y.min(), y.max())
print("Min/max z ;", z.min(), z.max())