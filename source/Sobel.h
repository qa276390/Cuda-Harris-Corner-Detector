#ifndef SOBEL_H
#define SOBEL_H_

/***********************
SERIAL SOBEL FILTER
************************/

/*
Sobel convolution masks
*/
int defaultMaskSize = 3;
float gxDefault[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};
float gyDefault[3][3] = {{-1, 0, 1}, {-2,0,2}, {-1,0,1 } };

/*
Sobel filter of image with gx and gy convolution masks
*/
Matrix sobelFilter( Matrix &image , Matrix &gx , Matrix &gy ){
	int nRows = image.getRows();
	int nCols = image.getCols();
	int maskSize = gx.getRows();

	Matrix newImage(nRows, nCols);

	int radio = maskSize/2;
	for( int i = 0 ; i < nRows ; ++i ){
		for( int j = 0 ; j < nCols ; ++j ){
			float sumGx = 0 , sumGy = 0;
			for( int x = -radio ; x <= radio ; ++x ){
				for( int y= -radio ; y <= radio ; ++y ){
					int nx = x + i;
					int ny = y + j;
					if( nx >= 0 && ny >= 0 && nx < nRows && ny < nCols ){
						float grayValueNeighbor = image(nx, ny);
						//Sobel mask for x-direction:
						float gxValue = gx(x + radio, y + radio);
					    //Sobel mask for x-direction:
					    float gyValue = gy(x + radio, y + radio);;
						sumGx += gxValue * grayValueNeighbor;
						sumGy += gyValue * grayValueNeighbor;
					}
				}
			}
			newImage(i, j ) = sqrt( sumGx * sumGx + sumGy * sumGy );
		}
	}
	return newImage;
}

Matrix sobelFilter(Matrix &image){
	Matrix gx(defaultMaskSize, defaultMaskSize);
	Matrix gy(defaultMaskSize, defaultMaskSize);

	for (int i = 0; i < defaultMaskSize; ++i){
		for (int j = 0; j < defaultMaskSize; ++j){
			gx(i, j) = gxDefault[i][j];
			gy(i, j) = gyDefault[i][j];
		}
	}
	Matrix sobel = sobelFilter(image, gx, gy);
	return sobel;
}

Matrix sobelFilter(float* image, int nRows, int nCols){

	Matrix newImage(nRows, nCols);

	int radio = defaultMaskSize / 2;
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			float sumGx = 0, sumGy = 0;
			for (int x = -radio; x <= radio; ++x){
				for (int y = -radio; y <= radio; ++y){
					int nx = x + i;
					int ny = y + j;
					if (nx >= 0 && ny >= 0 && nx < nRows && ny < nCols){
						float grayValueNeighbor = image[nx * nCols + ny];
						//Sobel mask for x-direction:
						float gxValue = gxDefault[ x + radio][y + radio];
						//%Sobel mask for x-direction:
						float gyValue = gyDefault[ x + radio][y + radio];
						sumGx += gxValue * grayValueNeighbor;
						sumGy += gyValue * grayValueNeighbor;
					}
				}
			}
			newImage(i, j) = sqrt(sumGx * sumGx + sumGy * sumGy);
		}
	}
	return newImage;
}

/*
Get default values as array
*/
float* getGradientX(){
	float* data = new float[9];
	for (int i = 0; i < defaultMaskSize; ++i){
		for (int j = 0; j < defaultMaskSize; ++j){
			data[i * defaultMaskSize + j] = gxDefault[i][j];
		}
	}
	return data;
}

float* getGradientY(){
	float* data = new float[9];
	for (int i = 0; i < defaultMaskSize; ++i){
		for (int j = 0; j < defaultMaskSize; ++j){
			data[i * defaultMaskSize + j] = gyDefault[i][j];
		}
	}
	return data;
}

int getGradientSize(){
	return defaultMaskSize;
}

#endif
