# Corner-Detector
Parallel implementation, in CUDA, of Corner Detection in images using Gauss and Sobel filters.

-------------------------------------------------------------------------------------------------

**Input:** RGB image in .ppm format.<br />
**Output:** Grayscale image with corners detected in .ppm format. Will be located in output folder.

Compilation and testing can be performed directly from makefile or with the following commands:

1. **Compilation:**<br />
`nvcc CornerDetector.cu -o CornerDetector`

2. **Execution:**<br />
`./CornerDetector <imagePath> <gaussMask=size> <tpb=threadsPerBlock> <sigma=doubleValue>`

**`<imagePath>`**<br />
Path where the image is located. <br />
Example: ../input/imageName.ppm

**`<gaussMask=size>`**<br />
Size of the gaussian mask.<br />
Default value = 7<br />
Example: gaussMask=7 -> will generate a 7x7 gaussian mask

**`<tpb=threadsPerBlock>`**<br />
Threads per block used in both Gauss and Sobel kernels.<br />
Default value=32 <br />
Example: tpb=16 -> 16x16, tpb=32-> 32x32,etc.

**`<sigma=doubleValue>`**<br />
Sigma value used in gaussianMask generation. <br />
Default value=1.5<br />
Example: sigma=0.5

Examples of execution:

`./CornerDetector ../input/image1.ppm gaussMask=9` <br />
`./CornerDetector ../input/image2.ppm gaussMask=7 tpb=32` <br />
`./CornerDetector ../input/image3.ppm gaussMask=5 tpb=16 sigma=0.5` <br />
