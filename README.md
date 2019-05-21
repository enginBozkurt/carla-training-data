# Carla Training Data
Generating training data from the Carla driving simulator in the KITTI dataset format

## KITTI dataset format

- /data/kitti
-     |- 2011_09_26
-        |- 2011_09_26_drive_0005_sync
  -          |- tracklet_labels.xml
  -          |- image_00
  -          |- image_01
  -          |- image_02
  -          |- image_03
  -          |- oxts
  -          |- velodyne_points

## Getting started and Setup
This project expects the carla folder to be inside this project  __i.e PythonClient/carla-training-data/carla__

## Running the client after running the Carla Server

```bash
## Running the client after running the Carla Server

$ python3 datageneration.py
```




