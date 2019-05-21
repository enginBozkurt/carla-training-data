
from carla.transform import Transform
import numpy as np

from datadescriptor import KittiDescriptor
from camera_utils import *
from constants import WINDOW_HEIGHT, WINDOW_WIDTH, MAX_RENDER_DEPTH_IN_METERS, MIN_VISIBLE_VERTICES_FOR_RENDER, VISIBLE_VERTEX_COLOR, OCCLUDED_VERTEX_COLOR, MIN_BBOX_AREA_IN_PX
from utils import degrees_to_radians
import logging


def bbox_2d_from_agent(agent, intrinsic_mat, extrinsic_mat, ext, bbox_transform, agent_transform, rotRP):  # rotRP expects point to be in Kitti lidar format
    """ Creates bounding boxes for a given agent and camera/world calibration matrices.
        Returns the modified image that contains the screen rendering with drawn on vertices from the agent """
    bbox = vertices_from_extension(ext)
    # transform the vertices respect to the bounding box transform
    bbox = bbox_transform.transform_points(bbox)
    # the bounding box transform is respect to the agents transform
    # so let's transform the points relative to it's transform
    bbox = agent_transform.transform_points(bbox)
    # agents's transform is relative to the world, so now,
    # bbox contains the 3D bounding box vertices relative to the world
    # Additionally, you can logging.info these vertices to check that is working
    # Store each vertex 2d points for drawing bounding boxes later
    vertices_pos2d = vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat)
    return vertices_pos2d


def calculate_occlusion_stats(image, vertices_pos2d, depth_map, draw_vertices=True):
    """ Draws each vertex in vertices_pos2d if it is in front of the camera 
        The color is based on whether the object is occluded or not.
        Returns the number of visible vertices and the number of vertices outside the camera.
    """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    for y_2d, x_2d, vertex_depth in vertices_pos2d:
        # if the point is in front of the camera but not too far away
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas((y_2d, x_2d)):
            is_occluded = point_is_occluded(
                (y_2d, x_2d), vertex_depth, depth_map)
            if is_occluded:
                vertex_color = OCCLUDED_VERTEX_COLOR
            else:
                num_visible_vertices += 1
                vertex_color = VISIBLE_VERTEX_COLOR
            if draw_vertices:
                draw_rect(image, (y_2d, x_2d), 4, vertex_color)
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def create_kitti_datapoint(agent, intrinsic_mat, extrinsic_mat, image, depth_image, player_measurements, rotRP, draw_3D_bbox=True):
    """ Calculates the bounding box of the given agent, and returns a KittiDescriptor which describes the object to be labeled """
    obj_type, agent_transform, bbox_transform, ext, location = transforms_from_agent(
        agent)

    if obj_type is None:
        logging.warning(
            "Could not get bounding box for agent. Object type is None")
        return image, None
    vertices_pos2d = bbox_2d_from_agent(
        agent, intrinsic_mat, extrinsic_mat, ext, bbox_transform, agent_transform, rotRP)

    depth_map = depth_image.data * 1000
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
        image, vertices_pos2d, depth_map, draw_vertices=draw_3D_bbox)

    midpoint = midpoint_from_agent_location(
        image, location, extrinsic_mat, intrinsic_mat)

    # At least N vertices has to be visible in order to draw bbox
    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER and num_vertices_outside_camera < MIN_VISIBLE_VERTICES_FOR_RENDER:
        midpoint[:3] = rotRP * midpoint[:3]

        bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
        area = calc_bbox2d_area(bbox_2d)
        if area < MIN_BBOX_AREA_IN_PX:
            logging.info("Filtered out bbox with too low area {}".format(area))
            return image, None
        if draw_3D_bbox:
            draw_3d_bounding_box(image, vertices_pos2d)
        from math import pi
        rotation_y = get_relative_rotation_y(agent, player_measurements) % pi

        datapoint = KittiDescriptor()
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(ext)
        datapoint.set_type(obj_type)
        datapoint.set_3d_object_location(midpoint)
        datapoint.set_rotation_y(rotation_y)
        return image, datapoint
    else:
        return image, None


def get_relative_rotation_y(agent, player_measurements):
    """ Returns the relative rotation of the agent to the camera in yaw
    The relative rotation is the difference between the camera rotation (on car) and the agent rotation"""
    # We only car about the rotation for the classes we do detection on
    if agent.vehicle.transform:
        rot_agent = agent.vehicle.transform.rotation.yaw
        rot_car = player_measurements.transform.rotation.yaw
        return degrees_to_radians(rot_agent - rot_car)


def vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat):
    """ Accepts a bbox which is a list of 3d world coordinates and returns a list 
        of the 2d pixel coordinates of each vertex. 
        This is represented as a tuple (y, x, d) where y and x are the 2d pixel coordinates
        while d is the depth. The depth can be used for filtering visible vertices.
    """
    vertices_pos2d = []
    for vertex in bbox:
        pos_vector = vertex_to_world_vector(vertex)
        # Camera coordinates
        transformed_3d_pos = proj_to_camera(pos_vector, extrinsic_mat)
        # 2d pixel coordinates
        pos2d = proj_to_2d(transformed_3d_pos, intrinsic_mat)

        # The actual rendered depth (may be wall or other object instead of vertex)
        vertex_depth = pos2d[2]
        x_2d, y_2d = WINDOW_WIDTH - pos2d[0],  WINDOW_HEIGHT - pos2d[1]
        vertices_pos2d.append((y_2d, x_2d, vertex_depth))
    return vertices_pos2d


def vertex_to_world_vector(vertex):
    """ Returns the coordinates of the vector in correct carla world format (X,Y,Z,1) """
    return np.array([
        [vertex[0, 0]],  # [[X,
        [vertex[0, 1]],  # Y,
        [vertex[0, 2]],  # Z,
        [1.0]  # 1.0]]
    ])


def vertices_from_extension(ext):
    """ Extraxts the 8 bounding box vertices relative to (0,0,0)
    https://github.com/carla-simulator/carla/commits/master/Docs/img/vehicle_bounding_box.png 
    8 bounding box vertices relative to (0,0,0)
    """
    return np.array([
        [ext.x,   ext.y,   ext.z],  # Top left front
        [- ext.x,   ext.y,   ext.z],  # Top left back
        [ext.x, - ext.y,   ext.z],  # Top right front
        [- ext.x, - ext.y,   ext.z],  # Top right back
        [ext.x,   ext.y, - ext.z],  # Bottom left front
        [- ext.x,   ext.y, - ext.z],  # Bottom left back
        [ext.x, - ext.y, - ext.z],  # Bottom right front
        [- ext.x, - ext.y, - ext.z]  # Bottom right back
    ])


def transforms_from_agent(agent):
    """ Returns the KITTI object type and transforms, locations and extension of the given agent """
    if agent.HasField('pedestrian'):
        obj_type = 'Pedestrian'
        agent_transform = Transform(agent.pedestrian.transform)
        bbox_transform = Transform(agent.pedestrian.bounding_box.transform)
        ext = agent.pedestrian.bounding_box.extent
        location = agent.pedestrian.transform.location
    elif agent.HasField('vehicle'):
        obj_type = 'Car'
        agent_transform = Transform(agent.vehicle.transform)
        bbox_transform = Transform(agent.vehicle.bounding_box.transform)
        ext = agent.vehicle.bounding_box.extent
        location = agent.vehicle.transform.location
    else:
        return (None, None, None, None, None)
    return obj_type, agent_transform, bbox_transform, ext, location


def calc_bbox2d_area(bbox_2d):
    """ Calculate the area of the given 2d bbox
    Input is assumed to be xmin, ymin, xmax, ymax tuple 
    """
    xmin, ymin, xmax, ymax = bbox_2d
    return (ymax - ymin) * (xmax - xmin)
