# CUDA-Harris-Corner-Detector
This project is a CUDA-acclerated Harris Corner Detector based on [Corner-Detector](https://github.com/jariasf/Corner-Detector).
We implemented the project using C/C++ with ***i7-7700*** and ***RTX 2080Ti***.

-------------------------------------------------------------------------------------------------
## Usage

### Folder and Data
Here is the directory tree structure for the project.
```

├── input
│   └── image.ppm           // input images(should be ppm format)
├── Makefile
├── output
│   └── ...                 // output images
├── README.md
└── source
    ├── CornerDetector.cu   // main part of the project
    ├── GaussFilter.cuh     // CUDA header file of gaussian filter 
    ├── Gauss.h             // header file of gaussian filter
    ├── Matrix.h            // header file of class Matrix
    ├── PPM.h               // header file to deal with portable pixmap format (PPM) image
    ├── SobelFilter.cuh     // CUDA header file of sobel filter 
    ├── Sobel.h             // header file of sobel filter
    ├── utils.h             // other utilities
    └── VectorOperation.cuh // CUDA header file of vector operation
```
We also implemented the shard memory and dynamic shared memory version of sobel filter.


### Prepare Images 
Put your images into the `./input` folder. Please note that the image must be `.ppm` format. With linux you can convert the image format using 
```
convert image.jpg image.ppm
```
### Start 
here is an example to run the code.
#### Compilation
you can just use `make` command, or
```
nvcc CornerDetector.cu -o CornerDetector
```

#### Execution
There are four parameter to assign.
```
./CornerDetector <imagePath> <gaussMask=size> <tpb=threadsPerBlock> <sigma=doubleValue>
```
example
```
./CornerDetector ./input/IMG_0125.ppm
```
