# Data

Data directories should be organized as follows and respective root paths need to be added to [datasets classes](../src/datasets). 


### SceneFlow
```
├── data
│   ├── SceneFlow 
│   │   ├── Driving
│   │   |   ├── disparity
│   │   |   |   ├── *.pfm
│   │   |   |   └── ...
│   │   |   ├── RGB_cleanpass
│   │   |   |   ├── left
│   │   |   |   |   ├── *.png
│   │   |   |   |   └── ...
│   │   |   |   ├── right
│   │   |   |   |   ├── *.png
│   │   |   |   |   └── ...
│   │   ├── Monkaa
│   │   |   ├── disparity
│   │   |   |   ├── *.pfm
│   │   |   |   └── ...
│   │   |   ├── RGB_cleanpass
│   │   |   |   ├── left
│   │   |   |   |   ├── *.png
│   │   |   |   |   └── ...
│   │   |   |   ├── right
│   │   |   |   |   ├── *.png
│   │   |   |   |   └── ...
│   │   └── FlyingThings3D
│   │   |   ├── disparity
│   │   |   |   ├── *.pfm
│   │   |   |   └── ...
│   │   |   ├── RGB_cleanpass
│   │   |   |   ├── left
│   │   |   |   |   ├── *.png
│   │   |   |   |   └── ...
│   │   |   |   ├── right
│   │   |   |   |   ├── *.png
│   │   |   |   |   └── ...
```

### KITTI 2012
```
├── data
│   ├── KITTI_2012
│   │   ├── data_stereo_flow
│   │   |   ├── training
│   │   |   |   ├── disp_nocc_0
│   │   |   |   |   ├── *.pfm
│   │   |   |   |   └── ...
│   │   |   |   ├── disp_nocc_1
│   │   |   |   |   ├── *.pfm
│   │   |   |   |   └── ...
│   │   |   |   ├── colored_0
│   │   |   |   |   ├── *.png
│   │   |   |   |   └── ...
│   │   |   |   ├── colored_1
│   │   |   |   |   ├── *.png
│   │   |   |   |   └── ...
```

### KITTI 2015
```
├── data
│   ├── KITTI_2015
│   │   ├── data_scene_flow
│   │   |   ├── training
│   │   |   |   ├── disp_nocc_0
│   │   |   |   |   ├── *.pfm
│   │   |   |   |   └── ...
│   │   |   |   ├── disp_nocc_1
│   │   |   |   |   ├── *.pfm
│   │   |   |   |   └── ...
│   │   |   |   ├── image_2
│   │   |   |   |   ├── *.png
│   │   |   |   |   └── ...
│   │   |   |   ├── image_3
│   │   |   |   |   ├── *.png
│   │   |   |   |   └── ...
```