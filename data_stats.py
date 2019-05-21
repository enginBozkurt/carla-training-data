

"""
This file is meant to generate statistics about a dataset generated in the KITTI format.
It prints by default:
- Average height and width of bounding boxes for each class
- The total number of objects of each class
- The average number of objects per scene per class
- The histogram of orientations of each class

"""
import numpy as np
import os
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.occlusion = int(data[2])
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        # location (x,y,z) in camera coord.
        self.t = (data[11], data[12], data[13])
        # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.ry = data[14]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' %
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' %
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' %
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' %
              (self.t[0], self.t[1], self.t[2], self.ry))


def print_stats(datapoints):
    avg_height_car = [x.h for x in datapoints if x.type == "Car"]
    avg_height_pedestrian = [
        x.h for x in datapoints if x.type == "Pedestrian"]
    print("Average height cars: ", sum(avg_height_car)/len(avg_height_car))
    
    bbox_pix_w = [x.xmax - x.xmin for x in datapoints]
    bbox_pix_h = [x.ymax - x.ymin for x in datapoints]
    from random import sample
    bbox_w, bbox_h = zip(*sample(list(zip(bbox_pix_w, bbox_pix_h)), 1000))
    plt.scatter(bbox_w, bbox_h, s=4)
    plt.xlabel("Bounding box width (px)")
    plt.ylabel("Bounding box height (px)")
    plt.savefig('test.png')
    
    
    """
    orientations = [x.ry for x in datapoints]
    n, bins, patches = plt.hist(
        orientations, 16, range=(0, 3.15))
    plt.xlabel('Orientation (rad)')
    plt.ylabel('Number of Cars')
    plt.yticks(range(0, 13000, 1000))
    plt.show()
    """

def read_data_dir(label_dirpath):
    datapoints = []
    cars_per_image = defaultdict(int)
    for filepath in os.listdir(label_dirpath):
        num_cars = 0
        for line in open(os.path.join(label_dirpath, filepath), "r").readlines():
            datapoint = Object3d(line)
            datapoints.append(datapoint)
            num_cars += 1
        cars_per_image[num_cars] += 1
    num_cars = len([x for x in datapoints if x.type == "Car"])

    print("Total number of cars: ", num_cars)

    print("Cars per image: ", cars_per_image)
    keys = cars_per_image.keys()
    values = cars_per_image.values()
    #plt.bar(keys, values)
    #plt.yticks(range(0, 13000, 1000))
    #plt.xlabel("Cars per image")
    #plt.ylabel("Number of images")
    #plt.show()
    print_stats(datapoints)


if __name__ == "__main__":
    read_data_dir(
        "/lhome/aasmunhb/Masteroppgave/CARS_20K/Carla/object/training/label_2")
