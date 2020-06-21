
#include <stdio.h>
#include <ctime>
#include <string>
#include <iostream>
#include <iomanip>
using namespace std;

#include "Matrix.h" //Matrix class
#include "utils.h" //Utils used in both Gauss and Sobel
#include "Gauss.h" //Serial Gauss
#include "Sobel.h" //Serial Sobel
#include "GaussFilter.cuh" //Parallel Gauss
#include "SobelFilter.cuh" //Parallel Sobel
#define DYNAMIC_GAUSS //Comment if you want to use static parameters in tiled convolution
#define DYNAMIC_SOBEL //Comment if you want to use static parameters in tiled convolution
float parallelHarrisCornerDetector(PPMImage* rgbimage, Matrix grayImage, Matrix gaussianKernel, string pathName, int numThreads);
float serialHarrisCornerDetector(PPMImage* rgbimage, Matrix grayImage, string pathName, int gaussKernelSize, double sigma);

int main(int argc, char* argv[]){
	if( argc == 1 ){
		cout<<"ERROR: Enter the name or path of an image file.\n"<<endl;
		return 0;
	}
	
	string imagePathName(argv[1]);
	
	//Default Values
	int gaussKernelSize = 7;
	int numThreads = 32; //Threads per block -- 32x32 or 16x16 or 8x8
	double sigma = 1.5;
	
	for( int i = 2 ; i < argc; ++i ){
		string key = getKey( string(argv[i]) );
		string val = getValue( string(argv[i]) );
		if( key == "gaussMask" ){
			gaussKernelSize = toInt(val);
		}else if( key == "sigma" ){
			sigma = toDouble(val);
		}else if( key == "tpb" ){
			numThreads = toInt(val);
		}
	}

	if (gaussKernelSize % 2 == 0){
		cout << "Mask has to be odd." << endl;
		return 0;
	}


	// Read Image
	cout << endl;
  PPMImage *image = readPPM(imagePathName.c_str());
	string outputFileName = getFileName(imagePathName,false);	
	cout << getFileName(imagePathName) << " -------------" << endl; 
  cout << endl;

	// Pre-Generate Gaussian Kernel
	Matrix grayImage = rgb2gray(image);
	Matrix gaussianKernel = createGaussianKernel(gaussKernelSize, sigma);




	// Harris Corner Detector
	double serialHarrisTime = serialHarrisCornerDetector(image, grayImage, outputFileName, gaussKernelSize, sigma);
	double parallelHarrisTime = parallelHarrisCornerDetector(image, grayImage, gaussianKernel, outputFileName, numThreads);


	cout << "Speedingup: " << serialHarrisTime / parallelHarrisTime << endl;



	//Free memory
	freeImage(image);

	return 0;
}


/*
	Harris Parallel Corner Detection using CUDA
*/
float parallelHarrisCornerDetector(PPMImage* rgbImage, Matrix grayImage, Matrix gaussianKernel, string pathName, int numThreads){

	clock_t startTime, endTime, clockTicksTaken;
	int imageWidth = grayImage.getCols(), imageHeight = grayImage.getRows();
	int imageSize = imageWidth * imageHeight;
	int kernelSize = gaussianKernel.getRows();
	float gaussTime = 0.0, sobelTime = 0.0, STime = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaError_t err;

	startTime = clock();
	//Arrays of CPU
	//float* result = allocateArray(imageSize);
	float* IxHost = allocateArray(imageSize);
	float* IyHost = allocateArray(imageSize);
	float* SxxHost = allocateArray(imageSize);
	float* SyyHost = allocateArray(imageSize);
	float* SxyHost = allocateArray(imageSize);
	float* grayImageArray = grayImage.toArray();
	float* gaussKernelArray = gaussianKernel.toArray();
	float* xGradient = getGradientX(), *yGradient = getGradientY(); //Used in Sobel

	//Arrays used in GPU
	float* kernelDevice; //Used in Gauss
	float* xGradDevice, *yGradDevice; //Used in Sobel
	float* imageDevice, *resultDevice; //Used in both Gauss and Sobel
	float* IxDevice, *IyDevice; //Used in both Gauss and Sobel
	float* IxxDevice, *IyyDevice, *IxyDevice; //Used in both Gauss and Sobel
	float* SxxDevice, *SyyDevice, *SxyDevice; //Used in both Gauss and Sobel

	//Allocate Device Memory
	cudaMalloc((void**)&imageDevice, imageSize * sizeof(float));
	cudaMalloc((void**)&resultDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&IxDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&IyDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&IxxDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&IyyDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&IxyDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&SxxDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&SyyDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&SxyDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&kernelDevice, kernelSize * kernelSize * sizeof(float));
	cudaMalloc((void**)&xGradDevice, SOBEL_MASK_SIZE * SOBEL_MASK_SIZE * sizeof(float));
	cudaMalloc((void**)&yGradDevice, SOBEL_MASK_SIZE * SOBEL_MASK_SIZE * sizeof(float));

	//Copy values from CPU to GPU
	cudaMemcpy(imageDevice, grayImageArray, imageSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernelDevice, gaussKernelArray, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(xGradDevice, xGradient, SOBEL_MASK_SIZE * SOBEL_MASK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yGradDevice, yGradient, SOBEL_MASK_SIZE * SOBEL_MASK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	
	endTime = clock();
	clockTicksTaken = endTime - startTime;
	double gpuInTime = clockTicksTaken / (double)CLOCKS_PER_SEC; 

	// Gaussian Blur to Remove Noise
	#ifdef DYNAMIC_GAUSS
		int tileWidth = numThreads - kernelSize + 1;
		int blockWidth = tileWidth + kernelSize - 1;
		dim3 dimGaussBlock(blockWidth, blockWidth);
		dim3 dimGaussGrid((imageWidth - 1) / tileWidth + 1, (imageHeight - 1) / tileWidth + 1);
	#else
		dim3 dimGaussBlock(GAUSS_BLOCK_WIDTH, GAUSS_BLOCK_WIDTH);
		dim3 dimGaussGrid((imageWidth - 1) / GAUSS_TILE_WIDTH + 1, (imageHeight - 1) / GAUSS_TILE_WIDTH + 1);
	#endif
	
	//Start recording GPU time
	cudaEventRecord(start, 0);
	#ifdef DYNAMIC_GAUSS
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (imageDevice, resultDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
	#else	
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (imageDevice, resultDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
	#endif	
	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("Gauss Error: %s\n", cudaGetErrorString(err));
	//GPU time recording finished
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//Copy GPU Time
	cudaEventElapsedTime(&gaussTime, start, stop);
	gaussTime *= 0.001;//Milliseconds to Seconds

	cudaDeviceSynchronize();

	// Sobel Filter to Compute Gx and Gy	

	//Tiled Sobel Filter
	#ifdef DYNAMIC_SOBEL
		int sobelTileWidth = numThreads - SOBEL_MASK_SIZE + 1;
		int sobelBlockWidth = sobelTileWidth + SOBEL_MASK_SIZE - 1;
		dim3 dimBlockSobel(sobelBlockWidth, sobelBlockWidth);
		dim3 dimGridSobel((imageWidth - 1) / sobelTileWidth + 1, (imageHeight - 1) / sobelTileWidth + 1);
	#else
		dim3 dimBlockSobel(SOBEL_BLOCK_WIDTH, SOBEL_BLOCK_WIDTH);
		dim3 dimGridSobel((imageWidth - 1) / SOBEL_TILE_WIDTH + 1, (imageHeight - 1) / SOBEL_TILE_WIDTH + 1);
	#endif
	
	// Copy result obtained from Gauss as new Image Data
	cudaMemcpy(imageDevice, resultDevice, imageSize * sizeof(float), cudaMemcpyDeviceToDevice);

	// Start recording GPU time
	cudaEventRecord(start, 0);
	#ifdef DYNAMIC_SOBEL
		DynamicSobelConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (imageDevice, IxDevice, imageWidth, imageHeight, xGradDevice, yGradDevice, sobelTileWidth, sobelBlockWidth, 0);
		DynamicSobelConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (imageDevice, IyDevice, imageWidth, imageHeight, xGradDevice, yGradDevice, sobelTileWidth, sobelBlockWidth, 1);
	#else	
		tiledSobelConvolution << < dimGridSobel, dimBlockSobel >> > (imageDevice, resultDevice, imageWidth, imageHeight, xGradDevice, yGradDevice);
	#endif	
	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("Sobel Error: %s\n", cudaGetErrorString(err));

	//GPU time recording finished
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//Copy GPU Time
	cudaEventElapsedTime(&sobelTime, start, stop);
	sobelTime *= 0.001; //Milliseconds to Seconds

	startTime = clock();
	//Copy Result from GPU to CPU
	cudaMemcpy(IxHost, IxDevice, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(IyHost, IyDevice, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
	endTime = clock();
	clockTicksTaken = endTime - startTime;
	double gpuOutTime = clockTicksTaken / (double)CLOCKS_PER_SEC; 


	startTime = clock();
	Matrix Ix = Matrix(arrayToMatrix(IxHost, imageHeight, imageWidth));
	Matrix Iy = Matrix(arrayToMatrix(IyHost, imageHeight, imageWidth));

	// Calculate S_(x^2), S_(y^2), S_(x*y)
	Matrix Ixx = Ix * Ix;
	Matrix Iyy = Iy * Iy;
	Matrix Ixy = Ix * Iy;
	float* IxxArray = Ixx.toArray(), *IyyArray = Iyy.toArray(), *IxyArray = Ixy.toArray();
	
	endTime = clock();
	clockTicksTaken = endTime - startTime;
	double MatrixTime = clockTicksTaken / (double)CLOCKS_PER_SEC; 

	startTime = clock(); // gpuInTime
	//Copy values from CPU to GPU
	cudaMemcpy(IxxDevice, IxxArray, imageSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(IyyDevice, IyyArray, imageSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(IxyDevice, IxyArray, imageSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernelDevice, gaussKernelArray, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(xGradDevice, xGradient, SOBEL_MASK_SIZE * SOBEL_MASK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yGradDevice, yGradient, SOBEL_MASK_SIZE * SOBEL_MASK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	endTime = clock();
	clockTicksTaken = endTime - startTime;
	gpuInTime += clockTicksTaken / (double)CLOCKS_PER_SEC; 
	
	//Start recording GPU time
	cudaEventRecord(start, 0);
	#ifdef DYNAMIC_GAUSS
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (IxxDevice, SxxDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (IyyDevice, SyyDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (IxyDevice, SxyDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
	#else
		// to-do
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (imageDevice, resultDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (imageDevice, resultDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (imageDevice, resultDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
	#endif	
	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("Gauss Error: %s\n", cudaGetErrorString(err));
	//GPU time recording finished
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//Copy GPU Time
	cudaEventElapsedTime(&STime, start, stop);
	STime *= 0.001;//Milliseconds to Seconds

	cudaDeviceSynchronize();
	
	startTime = clock();
	cudaMemcpy(SxxHost, SxxDevice, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(SyyHost, SyyDevice, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(SxyHost, SxyDevice, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
	
	endTime = clock();
	clockTicksTaken = endTime - startTime;
	gpuOutTime += clockTicksTaken / (double)CLOCKS_PER_SEC; 
	
	startTime = clock();
	Matrix Sxx = Matrix(arrayToMatrix(SxxHost, imageHeight, imageWidth));
	Matrix Syy = Matrix(arrayToMatrix(SyyHost, imageHeight, imageWidth));
	Matrix Sxy = Matrix(arrayToMatrix(SxyHost, imageHeight, imageWidth));

	freeArray(IxxArray);
	freeArray(IyyArray);
	freeArray(IxyArray);
	//freeArray(kernelArray);

	// Calculate R = det(M) - k(trace(M))^2 
  Matrix detM = Sxx * Syy - Sxy * Sxy;
	Matrix trM = Sxx + Syy;
	Matrix R = detM/(trM + 1E-8);

	endTime = clock();
	clockTicksTaken = endTime - startTime;
	MatrixTime += clockTicksTaken / (double)CLOCKS_PER_SEC; 


	// Save Image Result
	
	PPMImage* result = showHarrisResult(rgbImage, R);
	string showPath;
  #ifdef _WIN32
		showPath = ".\\output\\"+pathName+"_harris_show_gpu.ppm";
	#else
	  showPath = "./output/"+pathName+"_harris_show_gpu.ppm";  
	#endif
	writePPM(showPath.c_str(), result);
	freeImage(result);
	
	result = createImage(R.getRows(), R.getCols());
	Matrix normalized = normalize(R, 0, 255);
	matrixToImage(normalized, result);
  #ifdef _WIN32
		pathName = ".\\output\\"+pathName+"_harris_gpu.ppm";
	#else
	    pathName = "./output/"+pathName+"_harris_gpu.ppm";  
	#endif
	writePPM(pathName.c_str(), result);
	freeImage(result);

	cout << "GPU Host to Device Time: " << setprecision(3) << fixed << gpuInTime << "s." << endl;
	cout << "GPU Harris Gauss Time: " << setprecision(3) << fixed << gaussTime << "s." << endl;
	cout << "GPU Harris Sobel Time: " << setprecision(3) << fixed << sobelTime << "s." << endl;
	cout << "GPU Harris SG Time: " << setprecision(3) << fixed << STime << "s." << endl;
	cout << "GPU Harris Device to Host Time: " << setprecision(3) << fixed << gpuOutTime << "s." << endl;
	cout << "GPU Harris Matrix Time: " << setprecision(3) << fixed << MatrixTime << "s." << endl;
	cout << "GPU Harris Time: " << setprecision(3) << fixed << gaussTime + sobelTime + STime + gpuInTime + gpuOutTime + MatrixTime << "s." << endl;

	// Free Memory used in GPU and CPU
	cudaFree(xGradDevice);	cudaFree(yGradDevice); cudaFree(imageDevice);
	cudaFree(resultDevice); cudaFree(kernelDevice);
	freeArray(xGradient);	freeArray(yGradient);  //freeArray(result);
	freeArray(grayImageArray); freeArray(gaussKernelArray);
	
	return gaussTime + sobelTime + STime + gpuInTime + gpuOutTime + MatrixTime;
}

/*
CPU Harris Corner Detection
*/

float serialHarrisCornerDetector(PPMImage* rgbImage, Matrix grayImage,  string pathName, int gaussKernelSize,double sigma){
	PPMImage* result;
	Matrix normalized;
	clock_t startTime, endTime, clockTicksTaken;
	
	Matrix gaussianKernel = createGaussianKernel(gaussKernelSize, sigma); // could be different sigma 
	
	// Gaussian Blur to Remove Noise
	float* grayImageArray = grayImage.toArray(), *kernelArray = gaussianKernel.toArray();
	startTime = clock();
	Matrix noiseRemoval = GaussianBlur(grayImageArray, kernelArray, grayImage.getRows(),
									   grayImage.getCols(), gaussianKernel.getRows());
	endTime = clock();
	clockTicksTaken = endTime - startTime;
	double gaussTime = clockTicksTaken / (double)CLOCKS_PER_SEC; //gaussTime *= 1000.0;
	freeArray(grayImageArray);
	freeArray(kernelArray);


	startTime = clock();
	// Sobel Filter to Detect Corners
	float* noiseRemovalArray = noiseRemoval.toArray();
	Matrix Ix = sobelFilter(noiseRemovalArray, noiseRemoval.getRows(), noiseRemoval.getCols(), 0);
	Matrix Iy = sobelFilter(noiseRemovalArray, noiseRemoval.getRows(), noiseRemoval.getCols(), 1);
	freeArray(noiseRemovalArray);
	
	// Calculate S_(x^2), S_(y^2), S_(x*y)
	Matrix Ixx = Ix * Ix;
	Matrix Iyy = Iy * Iy;
	Matrix Ixy = Ix * Iy;
	float* IxxArray = Ixx.toArray(), *IyyArray = Iyy.toArray(), *IxyArray = Ixy.toArray();
	kernelArray = gaussianKernel.toArray();
	
	Matrix Sxx = GaussianBlur(IxxArray, kernelArray,Ixx.getRows(),
									   Ixx.getCols(), gaussianKernel.getRows());
	Matrix Syy = GaussianBlur(IyyArray, kernelArray, Iyy.getRows(),
									   Iyy.getCols(), gaussianKernel.getRows());
	Matrix Sxy = GaussianBlur(IxyArray, kernelArray, Ixy.getRows(),
									   Ixy.getCols(), gaussianKernel.getRows());

	freeArray(IxxArray);
	freeArray(IyyArray);
	freeArray(IxyArray);
	freeArray(kernelArray);

	// Calculate R = det(M) - k(trace(M))^2
  Matrix detM = Sxx * Syy - Sxy * Sxy;
	Matrix trM = Sxx + Syy;
	Matrix R = detM/(trM + 1E-8);

	
	endTime = clock();
	clockTicksTaken = endTime - startTime;
	double HarrisTime = clockTicksTaken / (double)CLOCKS_PER_SEC; //gaussTime *= 1000.0;
	
	// Save Image Result/////////////////
	/* 
		 to-do: showHarrisResult(img, harris_response)
	 */
	result = showHarrisResult(rgbImage, R);
	string showPath;
  #ifdef _WIN32
		showPath = ".\\output\\"+pathName+"_harris_show_cpu.ppm";
	#else
	  showPath = "./output/"+pathName+"_harris_show_cpu.ppm";  
	#endif
	writePPM(showPath.c_str(), result);
	freeImage(result);
	
	result = createImage(R.getRows(), R.getCols());
	normalized = normalize(R, 0, 255);
	matrixToImage(normalized, result);
  #ifdef _WIN32
		pathName = ".\\output\\"+pathName+"_harris_cpu.ppm";
	#else
	  pathName = "./output/"+pathName+"_harris_cpu.ppm";  
	#endif
	writePPM(pathName.c_str(), result);
	freeImage(result);
	


	cout << "CPU Harris Time: " << gaussTime + HarrisTime << "s." << endl;

	return gaussTime + HarrisTime;
}
