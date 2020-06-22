
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
#include "VectorOperation.cuh" //Vector Operation
//#define DYNAMIC_GAUSS //Comment if you want to use static parameters in tiled convolution
//#define DYNAMIC_SOBEL //Comment if you want to use static parameters in tiled convolution
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
	double parallelHarrisTime = parallelHarrisCornerDetector(image, grayImage, gaussianKernel, outputFileName, numThreads);
	double serialHarrisTime = serialHarrisCornerDetector(image, grayImage, outputFileName, gaussKernelSize, sigma);



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
	float gaussTime = 0.0, sobelTime = 0.0, STime = 0.0, vecTime = 0.0, vecTime0 = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaError_t err;
	
	int threadsPerBlock = 128;
	int blocksPerGrid = (imageSize + threadsPerBlock - 1)/threadsPerBlock;

	startTime = clock();
	//Arrays of CPU
	//float* result = allocateArray(imageSize);
	float* IxHost = allocateArray(imageSize);
	float* IyHost = allocateArray(imageSize);
	float* SxxHost = allocateArray(imageSize);
	float* SyyHost = allocateArray(imageSize);
	float* SxyHost = allocateArray(imageSize);
	float* RHost = allocateArray(imageSize);
	float* grayImageArray = grayImage.toArray();
	float* gaussKernelArray = gaussianKernel.toArray();
	float* xGradient = getGradientX(), *yGradient = getGradientY(); //Used in Sobel

	//Arrays used in GPU
	float* kernelDevice; //Used in Gauss
	float* xGradDevice, *yGradDevice; //Used in Sobel
	float* imageDevice, *resultDevice; //Used in both Gauss and Sobel
	float* IxDevice, *IyDevice; //Used in both Gauss and Sobel
	float* IxxDevice, *IyyDevice, *IxyDevice; //Used in both Gauss and Sobel
	float* SxxDevice, *SyyDevice, *SxyDevice; //
	float* tmp1Device, *tmp2Device, *detMDevice, *trMDevice, *ResponseDevice; //

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
	cudaMalloc((void**)&tmp1Device, imageSize* sizeof(float));
	cudaMalloc((void**)&tmp2Device, imageSize* sizeof(float));
	cudaMalloc((void**)&detMDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&trMDevice, imageSize* sizeof(float));
	cudaMalloc((void**)&ResponseDevice, imageSize* sizeof(float));
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

	/*
		 Gaussian Blur to Remove Noise
	*/
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

	
	/*
		 Sobel Filter to Compute Gx and Gy	
	*/
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
		tiledSobelConvolution << < dimGridSobel, dimBlockSobel >> > (imageDevice, IxDevice, imageWidth, imageHeight, xGradDevice, yGradDevice, 0);
		tiledSobelConvolution << < dimGridSobel, dimBlockSobel >> > (imageDevice, IyDevice, imageWidth, imageHeight, xGradDevice, yGradDevice, 1);
	#endif

	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("Sobel Error: %s\n", cudaGetErrorString(err));

	//GPU time recording finished
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//Copy GPU Time
	cudaEventElapsedTime(&sobelTime, start, stop);
	sobelTime *= 0.001; //Milliseconds to Seconds

	/*
		Calculate Ixx, Iyy and Ixy
	*/
	cudaEventRecord(start, 0);
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(IxDevice, IxDevice, IxxDevice, imageSize, 2); // 2: Multiplication(*)
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(IyDevice, IyDevice, IyyDevice, imageSize, 2); // 2: Multiplication(*)
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(IxDevice, IyDevice, IxyDevice, imageSize, 2); // 2: Multiplication(*)
	
	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("VecOpeartion 0 Error: %s\n", cudaGetErrorString(err));

	//GPU time recording finished
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//Copy GPU Time
	cudaEventElapsedTime(&vecTime0, start, stop);
	sobelTime *= 0.001; //Milliseconds to Seconds
	

	/*
	 	Caluculate Sxx, Syy and Sxy with Gaussian Filter
	 */
	cudaEventRecord(start, 0);
	#ifdef DYNAMIC_GAUSS
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (IxxDevice, SxxDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (IyyDevice, SyyDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
		DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (IxyDevice, SxyDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
	#else
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (IxxDevice, SxxDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (IyyDevice, SyyDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
		tiledConvolution << < dimGaussGrid, dimGaussBlock, 1 >> > (IxyDevice, SxyDevice, imageWidth, imageHeight, kernelDevice, kernelSize);
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




	/* Compute R = detM / trM*/
	cudaEventRecord(start, 0);
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(SxxDevice, SyyDevice, tmp1Device, imageSize, 2); // 2: Multiplication
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(SxyDevice, SxyDevice, tmp2Device, imageSize, 2); // 
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(tmp1Device, tmp2Device, detMDevice, imageSize, 1); // 0: Addition, detM = Sxx * Syy - Sxy * Sxy

	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(SxxDevice, SyyDevice, trMDevice, imageSize, 0); // 0: Addition, trM = Sxx + Syy
	
	VecOperation<<<blocksPerGrid, threadsPerBlock>>>(detMDevice, trMDevice, ResponseDevice, imageSize, 3); // 3: devision
	
	//VecOpResponse<<<blocksPerGrid, threadsPerBlock>>>(SxxDevice, SyyDevice, SxyDevice, ResponseDevice, imageSize);
	
	err = cudaGetLastError();
	if (err != cudaSuccess)	printf("VecOperation Error: %s\n", cudaGetErrorString(err));
	//GPU time recording finished
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//Copy GPU Time
	cudaEventElapsedTime(&vecTime, start, stop);
	vecTime *= 0.001;//Milliseconds to Seconds

	// GPU output
	startTime = clock();
	cudaMemcpy(RHost, ResponseDevice, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
	Matrix R = Matrix(arrayToMatrix(RHost, imageHeight, imageWidth));
	endTime = clock();
	clockTicksTaken = endTime - startTime;
	double gpuOutTime = clockTicksTaken / (double)CLOCKS_PER_SEC; 
	

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
	cout << "GPU Harris Matrix Time: " << setprecision(3) << fixed << vecTime << "s." << endl;
	cout << "GPU Harris Matrix 0 Time: " << setprecision(3) << fixed << vecTime0 << "s." << endl;
	cout << "GPU Harris Time: " << setprecision(3) << fixed << gaussTime + sobelTime + STime + gpuInTime + gpuOutTime + vecTime + vecTime0 << "s." << endl;

	// Free Memory used in GPU and CPU
	cudaFree(xGradDevice);	cudaFree(yGradDevice); cudaFree(imageDevice);
	cudaFree(resultDevice); cudaFree(kernelDevice); 
	cudaFree(IxxDevice);cudaFree(IyyDevice);cudaFree(IxyDevice);
	cudaFree(SxxDevice);cudaFree(SyyDevice);cudaFree(SxyDevice);
	freeArray(xGradient);	freeArray(yGradient);  //freeArray(result);
	freeArray(grayImageArray); freeArray(gaussKernelArray);
	
	return gaussTime + sobelTime + STime + gpuInTime + gpuOutTime + vecTime + vecTime0;
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
