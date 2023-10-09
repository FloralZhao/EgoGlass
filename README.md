# EgoGlass: Egocentric-View Human Pose Estimation From an Eyeglass Frame
This is the repo for our paper [EgoGlass: Egocentric-View Human Pose Estimation From an Eyeglass Frame (3DV 2021, oral)](https://www.computer.org/csdl/proceedings-article/3dv/2021/268800a032/1zWE6qypWak).

## EgoGlass dataset
### File formats
* BodycamCalib: Intrisics of body cameras (cameras on the eyeglasses, egocentric view)

* PGCalib: Intrinsics and extrinsics of PointGrey cameras (cameras on the wall, third-person view)

* Data for each subject
```
.                          
├── sitting                 
│   ├── skeletonData.txt  # 3D joint positions    
│   ├── bodyCamTrackFile.txt  # body cameras' tracking data
│   └── TrackingData  # images captured by body cameras
│   	└── AugmentedData  # augmented with random background
│           ├── BodyCam1
│           ├── BodyCam2 
│           └── ... 
│                  
├── standing                   
│   └── ...                 
└── walking 
    └── ...         
```

## Video
This [video](https://drive.google.com/file/d/1XcOdEVSEe1MOuYwuSQUGLPH4oFofnX1Z/view?usp=sharing) shows the results from consecutive frames, which demonstrates the temporal smoothness of our method.


