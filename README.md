# mip_calibrate
Auto-calibration for MIP robot

[![Build Status](https://travis-ci.org/arnaud-ramey/camera.svg)](https://travis-ci.org/arnaud-ramey/camera)


Licence
=======

LGPL v3 (GNU Lesser General Public License version 3).

How to install
==============

Dependencies from sources
-------------------------

Dependencies handling is based on the [wstool](http://wiki.ros.org/wstool) tool.
Run the following instructions:

```bash
$ sudo apt-get install python-wstool
$ roscd ; cd src
$ wstool init
$ wstool merge `rospack find mip_calibrate`/dependencies.rosinstall
$ wstool update
```

Dependencies included in the Ubuntu packages
--------------------------------------------

Please run the ```rosdep``` utility:

```bash
$ sudo apt-get install python-rosdep
$ sudo rosdep init
$ rosdep install mip_calibrate --ignore-src
```

Build package with Catkin
-------------------------

```bash
$ catkin_make --only-pkg-with-deps mip_calibrate
```
