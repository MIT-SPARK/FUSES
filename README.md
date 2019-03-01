README - FUSES: Fast Understanding via SEmidefinite Segmentation
======================================

What is FUSES?
------------

FUSES is a library of C++ classes for Fast Semantic Segmentation. Find overview of architecture here: 

Quickstart
----------

This repository links installation instructions to compile and run FUSES on the CityScapes Dataset for Ubuntu 16.04. Note that you cannot run the entire pipeline with Bonnet on Mac due to TensorRT being unavaliable on Mac.

Prerequisites (Find installation instructions below for all of the following):

- [GTSAM >= 4.0](#installation-of-gtsam)
- [OpenCV](#installation-of-opencv)
- [OpenGV](#installation-of-opengv)
- [HDF5 >= 1.8](#installation-of-hdf5)
- [YAML-cpp](#installation-of-yaml-cpp)
- [OpenGM >= 2.0](#installation-of-opengm)
- [Bonnet](#installation-of-bonnet)
- [CPLEX](#installation-of-cplex)

In the root library of the sdpSegmentation folder, execute:

```
#!bash
$ mkdir build
$ cd build
$ cmake ../
$ make check
$ make
$ make install
```

If there are errors about finding hdf5 include directory, use cmake-gui to set HDF5_INCLUDE_DIRS (e.g., on Linux computer, the default path is "/usr/include/hdf5/serial")

Running exampel with bonnet on Cityscapes dataset (in build folder):

```
$ ./bin/fuses-example-bonnet <path/to/Cityscapes/dataset> lindau ../bonnet/frozen_512 ../tests/data/fusesParameters.yaml
```

This repo contains a parser `src/include/FUSES/CityscapesParser.h` for the [Cityscapes](https://www.cityscapes-dataset.com) dataset. This parser will loop through images in the validation set. `<nameOfDataset>` can be set to munster, frankfurt, lindau, or all (meaning all three). [Bonnet](https://github.com/PRBonn/bonnet) provides three pre-trained models for this dataset and they are included in `bonnet` folder as `bonnet/frozen_*`.

KNOW ISSUES:
- you have to comment out "find_package(Bonnet)" (line 99) if you do not have Bonnet installed


## Installation of GTSAM

- on Linux 16.04:

1. Navigate to the link [here](https://bitbucket.org/gtborg/gtsam/overview/) and git clone https://bitbucket.org/gtborg/gtsam.git.
2. Switch into the `feature/ImprovementsIncrementalFilter` branch by doing git fetch && git checkout feature/improvementsIncrementalFilter 
3. In the gtsam folder do:

```
$ sudo apt-get install libboost-all-dev
$ sudo apt-get install cmake
$ mkdir build
$ cd build
$ cmake ..
$ make check (optional - to run unit tests)
$ make install

```

## Installation of OpenCV

- On Mac:
```
#!bash
$ brew install vtk
download and unzip opencv-3.4.1 from https://opencv.org/releases.html
download and unzip opencv_contrib-master from https://github.com/opencv/opencv_contrib
go to opencv-3.4.1
$ mkdir build
$ cd build
$ cmake -DWITH_VTK=On -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib-master>/modules/ ../
(typically: cmake -DWITH_VTK=On -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-master/modules/ ../)
$ sudo make install
```

- On Linux 16.04:

```
#!bash
$ sudo apt-get install libvtk5-dev   
$ sudo apt-get install libgtk2.0-dev 
$ sudo apt-get install pkg-config
download and unzip opencv-3.4.1 from https://opencv.org/releases.html
download and unzip opencv_contrib-master from https://github.com/opencv/opencv_contrib
go to opencv-3.4.1
$ mkdir build
$ cd build
$ cmake -DWITH_VTK=On -DOPENCV_EXTRA_MODULES_PATH=<path-to-opencv_contrib-master>/modules/ ../
$ sudo make -j8 install
$ sudo make -j8 test (optional - quite slow)
```

Notes: 
- `<path-to-opencv_contrib-master>` is the path to opencv_contrib-master folder. If it is placed in the same folder as `opencv-3.4.1`, the cmake command will be "cmake -DWITH_VTK=On -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-master/modules/ ..". 
- If there are errors installing opencv_contrib, delete all folders in `opencv_contrib_master/modules` except `ximgproc`. This is the only module needed for this project.

KNOWN ISSUES: This did not compile in Ubuntu 17.10. I had to apply several changes, including:
1) remove all instances of CV_OVERRIDE
2) remove all instances of CV_FINAL
3) comment all instances of CV_CheckTypes
4) add a return condition in the function Ptr<StereoMatcher> createRightMatcher(Ptr<StereoMatcher> matcher_left)

## Installation of OpenGV

1. Perform git clone https://github.com/laurentkneip/opengv (I did this in my "home/code/" folder)
2. (not needed in latest version) open CMakeLists.txt and set INSTALL_OPENGV to ON (this can be also done using cmake-gui)
3. using cmake-gui, set: the eigen version to the GTSAM one (for me: /Users/Luca/borg/gtsam/gtsam/3rdparty/Eigen). If you don't do so, very weird error appear (may be due to GTSAM and OpenGV using different versions of eigen!)
4. In the opengv folder do:

```
#!bash
$ mkdir build
$ cd build
$ cmake ../
$ sudo make -j8 install
$ make -j8 test
```

## Installation of YAML-cpp

- On Linux 16.04:

```
$ sudo apt-get install libyaml-cpp-dev

```
- On Mac:

```
$ brew install yaml-cpp
```

## Installation of HDF5

The installation of HDF5 packages varies by system. Below there are few examples: 

- On Ubuntu 16.04LTS:
```
$ sudo apt-get install libhdf5-10 libhdf5-cpp-11
```

Debian:
The library names of hdf5 on debian have a postfix "serial", therefore `-lhdf5` and `-lhdf5_hl` cannot be found. To resolve this problem, two symbolic links need to be created. You need to check the `/usr/lib/x86_64-linux-gnu` directory for the version used. The code below assumes the version to be v10 and need to be modified if other versions are used.
```
$ sudo ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.10 /usr/lib/x86_64-linux-gnu/libhdf5.so
$ sudo ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.10 /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
```

- On Ubuntu 17.10:
```
$ sudo apt-get install libhdf5-serial-dev
```
```
$ sudo ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so /usr/lib/x86_64-linux-gnu/libhdf5.so
$ sudo ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl_cpp.so /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
```
note: libraries get stored in /usr/lib/x86_64-linux-gnu/hdf5/serial (you may need to change cmake HDF5 directory)

- On Mac: 
```
$ brew install hdf5
```

## Installation of OpenGM

- On Linux 16.04: 

1. Clone openGM from here: https://github.com/opengm/opengm.git
2. Then, go in the openGM folder and execute:
```
#!bash
$ mkdir build
$ cd build
$ cmake ../
$ make -j8 
$ sudo make -j8 install
$ make -j8 test
```

## Installation of Bonnet

[Bonnet](https://github.com/PRBonn/bonnet) only works in Ubuntu 16.04 as TensorRT is not available for Ubuntu 17.10 or Mac. It is dependent on Cuda 9.0, Cudnn 7.0, and TensorRT 3.0 in that order. 

1. Install Cuda 9.0 and cudnn 7.0 by following the tutorial [here](https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e) (install corresponding nvidia driver & double check version from Bonnet installation) 
2. To Install TensorRT, go to ``deploy_cpp`` folder inside of Bonnet repository [here](https://github.com/PRBonn/bonnet) and install TensorRT 3.0 (install TensorRT 3.0.4 for Ubuntu 16.04 and CUDA 9.0 tar package, as this allows for more flexibility than the debian installation) [here](https://developer.nvidia.com/nvidia-tensorrt-download)

Choose where you want to install and unpack the tar file. This tar file will install everything into a directory
called TensorRT-3.x.x, where 3.x.x is your TensorRT version. 

```
tar xzvf TensorRT-3.x.x.Ubuntu-16.04.3.x86_64.cuda-9.0.cudnn7.0.tar.gz
```

Add the absolute path of TensorRT lib to the environment variable

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<eg:TensorRT-3.x.x/lib>
```
 
For more details, see installation guide when installing [here](https://developer.nvidia.com/nvidia-tensorrt-download)
 
3. To Run Bonnet inside of Docker:

Inside of the bonnet directory:
```
$ docker pull tano297/bonnet:cuda9-cudnn7-tf17-trt304
$ nvidia-docker build -t bonnet .
```

Put your model + image into another directory, e.g. /Documents

```
$ nvidia-docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/developer/.Xauthority -v /home/$USER/data:/shared --net=host --pid=host --ipc=host bonnet /bin/bash
```

Model and image are now put in /test

4. Testing Bonnet 
 
At this point, docker container should be running.
 
Go to “~/bonnet_wrkdir/deploy_cpp/src/ros/config” (or something like that), and edit the cnn_config.yaml file: model path points to “/test/frozen” [a different path, here you’re going to freeze your trained model into this directory]
 
[Only need to do this once] Setup dependencies for training as specified here: https://github.com/PRBonn/bonnet/tree/master/train_py
 
When you’re ready to freeze your graph: run the following under ``~/bonnet_wrkdir/train_py``

```
$ ./cnn_freeze.py -p /test/<name of pretrained model>-l /test/frozen
```
Run cnn_use.py or cnn_use_pb_tensorRT.py (for speed up) to to analyze the image.

```
$ ./cnn_use.py -l /test/frozen -p /test/<name of pretrained model>-i /test/<name of image>
$ ./cnn_use_pb_tensorRT.py -l /test/log -p /test/frozen -i /test/<name of image>
```
 
Results are in the /test/log
 

KNOW ISSUES:
- For cudnn installation, open the file: ``/usr/include/cudnn.h`` and change the line ``#include "driver_types.h"`` to ``#include <driver_types.h>`` if facing compilation issues. 
- If facing cmake issues after installing TensorRT 3.0, go back and delete nvidia-docker and rerun Nvidia-docker installation commands.

## Installation of CPLEX

- On Linux 16.04:

CPLEX is a solver that solves the discrete optimization problem exactly. The IBM ILOG CPLEX Optimization Studio v12.8 for Linux x86-64 (CNN06ML), can be installed [here](https://ibm.onthehub.com/WebStore/Account/SdmAuthorize.aspx?o=74551a04-4fe4-e811-810c-000d3af41938&ws=4af0af44-b75f-e611-9420-b8ca3a5db7a1&uid=509b40b1-4ee4-e811-810c-000d3af41938&sdm=0).

You will need to run the .bin file (using sudo). 

KNOWN ISSUES:
- If the CPLEX method (computeExact) is still not recognized when running the matlab script below, add the installation directory to your path in the MATLAB console as so:

```
addpath(“/opt/ibm/ILOG/CPLEX_Studio128/cplex/matlab/x86-64_linux”)
```

## Other Known Issues

1. After completing the procedure above, running `cmake ../` in this project does not work and complains about Eigen.
Therefore, (on mac) I had to do the following:

```
#!bash
$ brew install eigen (strangely, cmake was not able to find the GTSAM one, even after fixing the path, maybe a version issue)
$ brew install lapack (this may be unnecessary)
```

2. On Ubuntu, if an Eigen error is thrown after running `cmake ../` in this project, comment out the line 57 in the upper CMakeLists file (removing finding MKL package). 

3. On Ubuntu, change the path of TensorRT in ``sdpSegmentation/cmake_extensions/FindBonnet.cmake`` to where your TensorRT lies.


## Running Examples

### Running the Script in MATLAB

1. Create a folder called CityScapes in the same directory as the sdpSegmentation repository. In here, place two folders: gtFine_trainvaltest (downloaded [here](https://www.cityscapes-dataset.com/downloads/)) and leftImf8bittrainvaltest (downloaded [here](https://www.cityscapes-dataset.com/downloads/))
2. In order to run the code, you need to go into matlab/generateOptimizationResults.m and change the path for the variable “datasetPath” to the path where CityScapes lies
3. Run the pipeline in Matlab by running the script ``generateOptimizationResults.m``


### Running on RACECAR 

You can run the pipeline on the RACECAR platform [here](https://github.mit.edu/SPARK/fuses_ros)
