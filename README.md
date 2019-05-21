# Carla Training Data
Generating training data from the Carla driving simulator in the KITTI dataset format


![Screenshot_3](https://user-images.githubusercontent.com/30608533/58124632-5c87b380-7c17-11e9-907b-f7cfdbb697a7.jpg)



## KITTI dataset format

### An example of KITTI dataset format
![Screenshot_4](https://user-images.githubusercontent.com/30608533/58124233-6b219b00-7c16-11e9-9562-9504c5b24bad.jpg)


- Raw (unsynced+unrectified) and processed (synced+rectified) grayscale stereo sequences (0.5 Megapixels, stored in png format)
- Raw (unsynced+unrectified) and processed (synced+rectified) color stereo sequences (0.5 Megapixels, stored in png format)
- 3D Velodyne point clouds (100k points per frame, stored as binary float matrix)
- 3D GPS/IMU data (location, speed, acceleration, meta information, stored as text file)
- Calibration (Camera, Camera-to-GPS/IMU, Camera-to-Velodyne, stored as text file)
- 3D object tracklet labels (cars, trucks, trams, pedestrians, cyclists, stored as xml file)



## Getting started and Setup
This project expects the carla folder to be inside this project  __i.e PythonClient/carla-training-data/carla__

## Running the client after running the Carla Server

```bash
## Running the client after running the Carla Server

$ python3 datageneration.py
```


### References:

-  KITTI Vision Benchmark Suite    
[KITTI website](http://www.cvlibs.net/datasets/kitti/raw_data.php)
-  Karlsruhe Institute of Technology  
[Karlsruhe Institute of Technology website](http://www.kit.edu/english/index.php)
-  Toyota Technological Institute at Chicago 
[Toyota Technological Institute website](https://www.ttic.edu/)

