#ifndef GAUSS_H
#define GAUSS_H

/***********************
SERIAL GAUSSIAN FILTER
************************/

/*
Create a Gaussian kernel of sizexsize based on sigma
*/
Matrix createGaussianKernel( int size , double sigma ){
	Matrix kernel( size, size );
	double mean = size/2;
	double sum = 0.0;

	for( int x = 0 ; x < size ; ++x ){
		for( int y = 0 ; y < size ; ++y ){
			kernel( x, y ) = exp( -0.5 * ( ( x - mean ) * ( x - mean ) + ( y - mean ) * ( y - mean ) )/(sigma * sigma) );
            sum += kernel(x , y);
		}
	}
 
	for( int x = 0 ; x < size ; ++x ){
		for( int y = 0 ; y < size ; ++y ){
			kernel( x, y) /= sum;
		}
	}

	return kernel;
}

void createGaussianKernel( Matrix &kernel, int size , double sigma ){
	kernel = Matrix(size,size);

	double mean = size/2;
	double sum = 0.0;

	for( int x = 0 ; x < size ; ++x ){
		for( int y = 0 ; y < size ; ++y ){
			kernel( x, y ) = exp( -0.5 * ( ( x - mean ) * ( x - mean ) + ( y - mean ) * ( y - mean ) )/(sigma * sigma) );
            sum += kernel.at(x,y);
		}
	}

	for( int x = 0 ; x < size ; ++x ){
		for( int y = 0 ; y < size ; ++y ){
			kernel(x, y) /= sum;
		}
	}
}

/*
Gaussian Filter of the given image data with kernel
*/
Matrix GaussianBlur( Matrix &image, Matrix &kernel ){
	int nRows = image.getRows();
	int nCols = image.getCols();
	int size = kernel.getRows();
	int radio = size / 2;

	Matrix newImage(nRows, nCols);

	for( int i = 0 ; i < nRows ; ++i ){
		for( int j = 0 ; j < nCols ; ++j ){
			float sum = 0;
			for( int x = -radio ; x <= radio ; ++x ){
				for( int y= -radio ; y <= radio ; ++y ){
					int nx = x + i;
					int ny = y + j;
					if( nx >= 0 && ny >= 0 && nx < nRows && ny < nCols ){
						float grayValueNeighbor = image(nx,ny);
						sum += kernel(x + radio,y + radio) * grayValueNeighbor;
						//sum += kernel.at(x + radio, y + radio) * grayValueNeighbor;
					}
				}
			}
			newImage(i,j) = sum;
		}
	}
	return newImage;
}

Matrix GaussianBlur(float *image, float* kernel, int nRows, int nCols, int size){
	Matrix newImage(nRows, nCols);
	int radio = size / 2;

	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			float sum = 0;
			for (int x = -radio; x <= radio; ++x){
				for (int y = -radio; y <= radio; ++y){
					int nx = x + i;
					int ny = y + j;
					if (nx >= 0 && ny >= 0 && nx < nRows && ny < nCols){
						sum += kernel[(x + radio) * size + (y + radio)] * image[nx * nCols + ny];
					}
				}
			}
			newImage(i, j) = sum;
		}
	}
	return newImage;
}

#endif
