#ifndef _UTILS_H_
#define _UTILS_

#include <string>
#include <sstream>
#include "PPM.h"
#include <fstream>

int getLocalIndex(int x, int y, int width){
	return x * width + y;
}

double getGrayFromRGB( double R , double G , double B ){
	return 0.299 * R + 0.587 * G + 0.114 * B;
}

/*
Conversion of an image from RBG to Gray
*/
Matrix rgb2gray(PPMImage* image){
	int nRows = image->y;
	int nCols = image->x;
	double r, g, b;
	Matrix gray(nRows, nCols);
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			int index = i * nCols + j;
			r = image->data[index].red;
			g = image->data[index].green;
			b = image->data[index].blue;
			gray(i, j) = getGrayFromRGB(r, g, b);
		}
	}
	return gray;
}

/*
Convert an Image object to Matrix object
*/
void matrixToImage(Matrix &imageData, PPMImage *image){
	int nRows = imageData.getRows();
	int nCols = imageData.getCols();
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			float value = imageData(i, j);
			image->data[i * nCols + j].red = value;
			image->data[i * nCols + j].green = value;
			image->data[i * nCols + j].blue = value;
		}
	}
}

/*
Normalize image data to values between [min_value,max_value]
*/
Matrix normalize(Matrix &image, const int min_value, const int max_value) {
	const float a = min_value < max_value ? min_value : max_value, b = min_value < max_value ? max_value : min_value;
	float min = image.getMin();
	float max = image.getMax();
	int nRows = image.getRows();
	int nCols = image.getCols();
	Matrix newImage(nRows, nCols);
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			newImage(i, j) = (image.at(i, j) - min) / (max - min)  * (b - a) + a;
			//newImage(i, j) = (image(i, j) - max) / (max - min)  * (b - a) + b;
		}
	}
	return newImage;
}

/*
Convert array to Matrix object
*/
Matrix arrayToMatrix(float* img, int rows , int cols ){
	Matrix data(rows, cols);
	for (int i = 0; i < rows; ++i){
		for (int j = 0; j < cols; ++j){
			data(i, j) = img[i * cols + j];
		}
	}
	return data;
}

/*
	Get name of a normal file:
    input /fileName.extension
    output fileName or fileName.extension
*/
string getFileName(const string& fileName, bool extension=true) {
   char sep = '/';
#ifdef _WIN32
   sep = '\\';
#endif
   size_t i = fileName.rfind(sep, fileName.length());
   if (i != string::npos) {
      string nameExt = fileName.substr(i+1, fileName.length() - i);
      size_t j = nameExt.rfind('.', nameExt.length() );
      if( j != string::npos && !extension  )
      	return(nameExt.substr(0, j));
	  return nameExt;
   }
   return fileName;
}

string toStr(int n){
    string s;
    ostringstream buffer;
    buffer<<n;
    s=buffer.str();
    return s;
}


int toInt(string str){
    int n;
    istringstream buffer(str);
    buffer>>n;
    return n;
}

double toDouble(string str){
    double n;
    istringstream buffer(str);
    buffer>>n;
    return n;
}

string getKey(string input){
    return input.substr(0, input.find("="));
}
string getValue(string input){
    return input.substr(input.find("=") + 1);
}

/*
Auxiliar functions to allocate and free memory
*/
float** allocateMatrix(int rows, int cols){
	float** data = new float*[rows];
    for( int i = 0 ; i < rows ; ++i )
		data[i] = new float[i];
    return data;
}

float* allocateArray(int size){
	float* data = new float[size];
	return data;
}

void freeArray(float* data){
	delete[] data;
}

void freeMatrix(int size, float**kernel){
	for( int i = 0 ; i < size ; ++i )
		free(kernel[i]);
	free(kernel);
}

#endif
