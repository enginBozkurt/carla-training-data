import time
import colorsys
import math
import numpy as np


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


def rand_color(seed):
    """Return random color based on a seed"""
    random.seed(seed)
    col = colorsys.hls_to_rgb(random.random(), random.uniform(.2, .8), 1.0)
    return (int(col[0]*255), int(col[1]*255), int(col[2]*255))


def vector3d_to_array(vec3d):
    return np.array([vec3d.x, vec3d.y, vec3d.z])


def degrees_to_radians(degrees):
    return degrees * math.pi / 180
