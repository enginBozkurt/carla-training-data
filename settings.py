from constants import *
from carla.settings import CarlaSettings
from carla import sensor
import random
import numpy as np
import math

from carla.transform import Transform, Rotation, Scale


def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(0, 0.0, CAMERA_HEIGHT_POS)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)

    lidar = sensor.Lidar('Lidar32')
    lidar.set_position(0, 0.0, LIDAR_HEIGHT_POS)
    lidar.set_rotation(0, 0, 0)
    lidar.set(
        Channels=40,
        Range=MAX_RENDER_DEPTH_IN_METERS,
        PointsPerSecond=720000,
        RotationFrequency=10,
        UpperFovLimit=7,
        LowerFovLimit=-16)
    settings.add_sensor(lidar)
    """ Depth camera for filtering out occluded vehicles """
    depth_camera = sensor.Camera('DepthCamera', PostProcessing='Depth')
    depth_camera.set(FOV=90.0)
    depth_camera.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    depth_camera.set_position(0, 0, CAMERA_HEIGHT_POS)
    depth_camera.set_rotation(0, 0, 0)
    settings.add_sensor(depth_camera)
    # (Intrinsic) K Matrix
    # | f 0 Cu
    # | 0 f Cv
    # | 0 0 1
    # (Cu, Cv) is center of image
    k = np.identity(3)
    k[0, 2] = WINDOW_WIDTH_HALF
    k[1, 2] = WINDOW_HEIGHT_HALF
    f = WINDOW_WIDTH / \
        (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    camera_to_car_transform = camera0.get_unreal_transform()
    lidar_to_car_transform = lidar.get_transform(
    ) * Transform(Rotation(yaw=90), Scale(z=-1))
    return settings, k, camera_to_car_transform, lidar_to_car_transform
