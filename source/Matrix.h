#ifndef MATRIX_H
#define MATRIX_H_
#include "stdlib.h"
/***
Matrix class
***/
typedef float T;

class Matrix{

    private:

        T** data; //Data of matrix
        int rows, cols; //The number of rows and columns
        void allocateMatrix();
        void setData(T value);
        void setData(T** newData);
    public:

        Matrix();
        Matrix(Matrix &matrix);
        Matrix( int rows, int cols );
        Matrix( int rows, int cols, T value );
        Matrix& operator=( const Matrix& rhs );
        Matrix operator-( const Matrix& rhs );
        Matrix operator+( const Matrix& rhs );
        Matrix operator+( float rhx );
        Matrix operator*( const Matrix& rhs );
        Matrix operator/( const Matrix& rhs );
        Matrix(const Matrix &other);
        Matrix( int rows, int cols, T** values );
        ~Matrix();
        inline void consolePrint();
        T& operator ()(int row, int column);
				T getMin();
				T getMax();
				T getSum();
				T getMean();
				T* toArray();
        inline void setValue( int row, int col, T value );
        int getRows();
        int getCols();
        void setRows(int row);
        void setCols(int col);
				inline T at(int row, int col);
				
};

void Matrix::consolePrint(){
    for( int i = 0 ; i < rows ; ++i ){
        for( int j = 0 ; j < cols ; ++j ){
            printf("%lf ", data[i][j]);
        }
        printf("\n");
    }
}

Matrix::Matrix( int rows, int cols, T** values ){
    this -> rows = rows;
    this -> cols = cols;
    allocateMatrix();
    setData(values);
}

Matrix::Matrix(int row , int col){
    rows = row;
    cols = col;
    allocateMatrix();
}

Matrix::Matrix(int row , int col, T value){
    this->rows = rows;
    this->cols = cols;
    allocateMatrix();
    setData(value);
}

Matrix::Matrix(Matrix &other){
	rows = other.rows;
	cols = other.cols;
	allocateMatrix();
	setData(other.data);
}

Matrix::Matrix(const Matrix &other){
    rows = other.rows;
    cols = other.cols;
    allocateMatrix();
    setData(other.data);
}

Matrix::Matrix(){

}

Matrix& Matrix::operator=( const Matrix& other ) { // assign data
    rows = other.rows;
    cols = other.cols;
    allocateMatrix();
    setData(other.data);
    return *this;
}

Matrix Matrix::operator*( const Matrix& other ) { // element-wise multiplication
    if( rows!=other.rows || cols != other.cols){
			return *(new Matrix());
		}
		Matrix result = Matrix(rows, cols);
    
		for( int i = 0 ; i < rows; ++i ){
        for( int j = 0 ; j < cols ; ++j ){
            result.setValue(i, j, data[i][j] * other.data[i][j]);
        }
		}
    
    return result;
}

Matrix Matrix::operator+( const Matrix& other ) { // element-wise addition
    if( rows!=other.rows || cols != other.cols){
			return *(new Matrix());
		}
		Matrix result = Matrix(rows, cols);
    
		for( int i = 0 ; i < rows; ++i ){
        for( int j = 0 ; j < cols ; ++j ){
            result.setValue(i, j, data[i][j] + other.data[i][j]);
        }
		}
    return result;
}

Matrix Matrix::operator+( float rhx ) { // element-wise addition
		Matrix result = Matrix(rows, cols);
    
		for( int i = 0 ; i < rows; ++i ){
        for( int j = 0 ; j < cols ; ++j ){
            result.setValue(i, j, data[i][j] + rhx);
        }
		}
    return result;
}
Matrix Matrix::operator-( const Matrix& other ) { // element-wise multiplication
    if( rows!=other.rows || cols != other.cols){
			return *(new Matrix());
		}
		Matrix result = Matrix(rows, cols);
    
		for( int i = 0 ; i < rows; ++i ){
        for( int j = 0 ; j < cols ; ++j ){
            result.setValue(i, j, data[i][j] - other.data[i][j]);
        }
		}
    return result;
}

Matrix Matrix::operator/( const Matrix& other ) { // element-wise multiplication
    if( rows!=other.rows || cols != other.cols){
			return *(new Matrix());
		}
		Matrix result = Matrix(rows, cols);
    
		for( int i = 0 ; i < rows; ++i )
        for( int j = 0 ; j < cols ; ++j ){
            result.setValue(i, j, data[i][j] / other.data[i][j]);
        }
    
    return result;
}
Matrix::~Matrix(){
    if( data != NULL ){
		for( int i = 0 ; i < rows ; ++i ){
            delete[] data[i];
		}
		delete[] data;
    }
}

void Matrix::allocateMatrix(){
    data = new T*[ rows ];
    for( int i = 0 ; i < rows ; ++i )
        data[i] = new T[cols];
}

void Matrix::setData( T value ){
    for( int i = 0 ; i < rows; ++i )
        for( int j = 0 ; j < cols ; ++j )
            data[i][j] = value;
}

void Matrix::setData( T** newData ){
    for( int i = 0 ; i < rows; ++i )
        for( int j = 0 ; j < cols ; ++j ){
            data[i][j] = newData[i][j];
        }
}

T& Matrix::operator ()(int row, int column){
        return data[row][column];
}

inline void Matrix::setValue( int row, int col, T value ){
    data[ row ][ col ] = value;
}

int Matrix::getRows(){
    return rows;
}

int Matrix::getCols(){
    return cols;
}

void Matrix::setRows(int row){
    rows = row;
}

void Matrix::setCols(int col){
    cols = col;
}

T Matrix::getMin(){
	T min = data[0][0];
	for (int i = 0; i < rows; ++i){
		for (int j = 0; j < cols; ++j){
			if (data[i][j] < min)
				min = data[i][j];
		}
	}
	return min;
}

T Matrix::getMax(){
	T max = data[0][0];
	for (int i = 0; i < rows; ++i){
		for (int j = 0; j < cols; ++j){
			if (data[i][j] > max)
				max = data[i][j];
		}
	}
	return max;
}

T Matrix::getSum(){
	T sum = data[0][0];
	for (int i = 0; i < rows; ++i){
		for (int j = 1; j < cols; ++j){
				sum += data[i][j];
		}
	}
	return sum;
}
T Matrix::getMean(){
	return getSum() / (rows * cols);
}
T* Matrix::toArray(){
	T* array = new T[ rows * cols ];
	for (int i = 0; i < rows; ++i){
		for (int j = 0; j < cols; ++j){
			array[i * cols + j] = data[i][j];
		}
	}
	return array;
}

inline T Matrix::at(int row, int col){
	return data[row][col];
}

#endif
