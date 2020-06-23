# CUDA-Harris-Corner-Detector
This project is a CUDA-acclerated Harris Corner Detector based on [Corner-Detector](https://github.com/jariasf/Corner-Detector).
We implemented the project using C/C++ with ***i7-7700*** and ***RTX 2080Ti***.

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
## Results
To compare the perfomance between CPU and GPU, we perform our Harris Corner Detector on a  image of size *3648x5472*, and the comparison results shown in *Table 1*. We could see that the GPU version is much faster the CPU version. 

||CPU|GPU-Shared-Memory|GPU-Dynamic-Shared-Memory|
|-|-|-|-|
|Running Time|17.378s|0.246s|0.247s|
|Difference| - | 0 | 0 |

The output image shown below. The red spot are Harris Corner Response.
<table>
<tr>
<th><img src="https://github.com/qa276390/Cuda_Harris_Corner_Detector/blob/master/output/IMG_0125_harris_show_gpu.jpg" /><br>Result from IMG_0125.ppm</th>
<th><img src="https://github.com/qa276390/Cuda_Harris_Corner_Detector/blob/master/output/IMG_4486_harris_show_gpu.jpg" /><br>Result from IMG_4486.ppm</th>
</tr>
</table>

## Reference
- [Digital Visual Effects, Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/20spring/assignments/proj1/)
- [Introduction to Cuda Parallel Programming, Ting-Wai Chiu](https://nol2.aca.ntu.edu.tw/nol/coursesearch/print_table.php?course_id=222%20D3160&class=&dpt_code=2220&ser_no=35335&semester=106-2&lang=CH)
- [Corner-Detector, jariasf](https://github.com/jariasf/Corner-Detector)
