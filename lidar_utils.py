

from camera_utils import proj_to_camera, proj_to_2d, draw_rect
from datageneration import WINDOW_WIDTH, WINDOW_HEIGHT
import numpy as np


def project_point_cloud(image, point_cloud, intrinsic_mat, draw_each_nth=5):
    """ Projects the lidar measurements onto the screen and draws them.
    Since the points are just X,Y,Z coordinates relative to the lidar, this can be done by a simple projection. 
    Note that this assumes that the camera and the lidar have exactly the same position and rotation!!
    """
    num_samples, dim = point_cloud.shape
    assert dim == 3, "Point cloud should have shape (?, 3) (X, Y, Z)"
    pos2d = np.dot(intrinsic_mat, point_cloud.T)
    for j in range(0, num_samples, draw_each_nth):
        cur_pos2d = pos2d[..., j]
        cur_pos2d = np.array([
            cur_pos2d[0] / cur_pos2d[2],
            cur_pos2d[1] / cur_pos2d[2],
            cur_pos2d[2]
        ])
        depth = cur_pos2d[2]

        if 1000 > depth > 0:
            x_2d = WINDOW_WIDTH - cur_pos2d[0]
            y_2d = WINDOW_HEIGHT - cur_pos2d[1]
            draw_rect(image, (y_2d, x_2d), int(20/depth))

    return image
