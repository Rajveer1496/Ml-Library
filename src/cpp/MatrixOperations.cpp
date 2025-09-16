#include <jni.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "core_utils_MatrixOperations.h"

class Matrix {
private:
    std::vector<std::vector<double>> data;
    int rows, cols;

public:
    // constructors
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    Matrix(const std::vector<std::vector<double>>& mat) : data(mat) {
        rows = mat.size();
        cols = rows > 0 ? mat[0].size() : 0;
    }

    // accessors
    double& operator()(int i, int j) { return data[i][j]; }
    const double& operator()(int i, int j) const { return data[i][j]; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }

    // matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("dimensions must match for addition");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = data[i][j] + other(i, j);
            }
        }
        return result;
    }

    // matrix subtraction
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("dimensions must match for subtraction");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = data[i][j] - other(i, j);
            }
        }
        return result;
    }

    // matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("invalid dimensions for multiplication");
        }
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    result(i, j) += data[i][k] * other(k, j);
                }
            }
        }
        return result;
    }

    // scalar multiplication
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = data[i][j] * scalar;
            }
        }
        return result;
    }

    // matrix transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }

    // determinant calculation using lu decomposition
    double determinant() const {
        if (rows != cols) {
            throw std::invalid_argument("determinant only for square matrices");
        }

        if (rows == 1) return data[0][0];
        if (rows == 2) return data[0][0] * data[1][1] - data[0][1] * data[1][0];

        Matrix temp = *this;
        double det = 1.0;

        for (int i = 0; i < rows; i++) {
            // find pivot
            int pivot = i;
            for (int j = i + 1; j < rows; j++) {
                if (std::abs(temp(j, i)) > std::abs(temp(pivot, i))) {
                    pivot = j;
                }
            }

            if (std::abs(temp(pivot, i)) < 1e-10) return 0.0;

            if (pivot != i) {
                std::swap(temp.data[i], temp.data[pivot]);
                det *= -1;
            }

            det *= temp(i, i);

            for (int j = i + 1; j < rows; j++) {
                double factor = temp(j, i) / temp(i, i);
                for (int k = i; k < cols; k++) {
                    temp(j, k) -= factor * temp(i, k);
                }
            }
        }
        return det;
    }

    // matrix inverse using gauss-jordan elimination
    Matrix inverse() const {
        if (rows != cols) {
            throw std::invalid_argument("inverse only for square matrices");
        }

        double det = determinant();
        if (std::abs(det) < 1e-10) {
            throw std::invalid_argument("matrix is singular");
        }

        // create augmented matrix [A|I]
        Matrix augmented(rows, 2 * cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                augmented(i, j) = data[i][j];
                augmented(i, j + cols) = (i == j) ? 1.0 : 0.0;
            }
        }

        // gauss-jordan elimination
        for (int i = 0; i < rows; i++) {
            // find pivot
            int pivot = i;
            for (int j = i + 1; j < rows; j++) {
                if (std::abs(augmented(j, i)) > std::abs(augmented(pivot, i))) {
                    pivot = j;
                }
            }

            if (pivot != i) {
                std::swap(augmented.data[i], augmented.data[pivot]);
            }

            // scale pivot row
            double pivotVal = augmented(i, i);
            for (int j = 0; j < 2 * cols; j++) {
                augmented(i, j) /= pivotVal;
            }

            // eliminate column
            for (int j = 0; j < rows; j++) {
                if (j != i) {
                    double factor = augmented(j, i);
                    for (int k = 0; k < 2 * cols; k++) {
                        augmented(j, k) -= factor * augmented(i, k);
                    }
                }
            }
        }

        // extract inverse matrix
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = augmented(i, j + cols);
            }
        }
        return result;
    }

    // matrix trace (sum of diagonal elements)
    double trace() const {
        if (rows != cols) {
            throw std::invalid_argument("trace only for square matrices");
        }
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            sum += data[i][i];
        }
        return sum;
    }

    // check if matrix is symmetric
    bool isSymmetric() const {
        if (rows != cols) return false;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (std::abs(data[i][j] - data[j][i]) > 1e-10) {
                    return false;
                }
            }
        }
        return true;
    }

    // matrix rank using gaussian elimination
    int rank() const {
        Matrix temp = *this;
        int rank = 0;

        for (int col = 0, row = 0; col < cols && row < rows; col++) {
            // find pivot
            int pivot = row;
            for (int i = row + 1; i < rows; i++) {
                if (std::abs(temp(i, col)) > std::abs(temp(pivot, col))) {
                    pivot = i;
                }
            }

            if (std::abs(temp(pivot, col)) < 1e-10) continue;

            if (pivot != row) {
                std::swap(temp.data[row], temp.data[pivot]);
            }

            rank++;

            for (int i = row + 1; i < rows; i++) {
                double factor = temp(i, col) / temp(row, col);
                for (int j = col; j < cols; j++) {
                    temp(i, j) -= factor * temp(row, j);
                }
            }
            row++;
        }
        return rank;
    }

    // convert to 1d array for jni
    std::vector<double> toArray() const {
        std::vector<double> result;
        result.reserve(rows * cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.push_back(data[i][j]);
            }
        }
        return result;
    }
};

// convert jni array to matrix
Matrix jArrayToMatrix(JNIEnv* env, jdoubleArray jArray, jint rows, jint cols) {
    jdouble* elements = env->GetDoubleArrayElements(jArray, NULL);
    Matrix matrix(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix(i, j) = elements[i * cols + j];
        }
    }

    env->ReleaseDoubleArrayElements(jArray, elements, JNI_ABORT);
    return matrix;
}

// convert matrix to jni array
jdoubleArray matrixToJArray(JNIEnv* env, const Matrix& matrix) {
    std::vector<double> data = matrix.toArray();
    jdoubleArray result = env->NewDoubleArray(data.size());
    env->SetDoubleArrayRegion(result, 0, data.size(), data.data());
    return result;
}

extern "C" {

// matrix addition
JNIEXPORT jdoubleArray JNICALL 
Java_core_utils_MatrixOperations_addMatrices(JNIEnv* env, jobject, 
    jdoubleArray matA, jint rowsA, jint colsA, jdoubleArray matB, jint rowsB, jint colsB) {
    try {
        Matrix a = jArrayToMatrix(env, matA, rowsA, colsA);
        Matrix b = jArrayToMatrix(env, matB, rowsB, colsB);
        Matrix result = a + b;
        return matrixToJArray(env, result);
    } catch (const std::exception& e) {
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return nullptr;
    }
}

// matrix subtraction
JNIEXPORT jdoubleArray JNICALL 
Java_core_utils_MatrixOperations_subtractMatrices(JNIEnv* env, jobject, 
    jdoubleArray matA, jint rowsA, jint colsA, jdoubleArray matB, jint rowsB, jint colsB) {
    try {
        Matrix a = jArrayToMatrix(env, matA, rowsA, colsA);
        Matrix b = jArrayToMatrix(env, matB, rowsB, colsB);
        Matrix result = a - b;
        return matrixToJArray(env, result);
    } catch (const std::exception& e) {
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return nullptr;
    }
}

// matrix multiplication
JNIEXPORT jdoubleArray JNICALL 
Java_core_utils_MatrixOperations_multiplyMatrices(JNIEnv* env, jobject, 
    jdoubleArray matA, jint rowsA, jint colsA, jdoubleArray matB, jint rowsB, jint colsB) {
    try {
        Matrix a = jArrayToMatrix(env, matA, rowsA, colsA);
        Matrix b = jArrayToMatrix(env, matB, rowsB, colsB);
        Matrix result = a * b;
        return matrixToJArray(env, result);
    } catch (const std::exception& e) {
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return nullptr;
    }
}

// scalar multiplication
JNIEXPORT jdoubleArray JNICALL 
Java_core_utils_MatrixOperations_scalarMultiply(JNIEnv* env, jobject, 
    jdoubleArray matrix, jint rows, jint cols, jdouble scalar) {
    try {
        Matrix a = jArrayToMatrix(env, matrix, rows, cols);
        Matrix result = a * scalar;
        return matrixToJArray(env, result);
    } catch (const std::exception& e) {
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return nullptr;
    }
}

// matrix transpose
JNIEXPORT jdoubleArray JNICALL 
Java_core_utils_MatrixOperations_transpose(JNIEnv* env, jobject, 
    jdoubleArray matrix, jint rows, jint cols) {
    try {
        Matrix a = jArrayToMatrix(env, matrix, rows, cols);
        Matrix result = a.transpose();
        return matrixToJArray(env, result);
    } catch (const std::exception& e) {
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return nullptr;
    }
}

// matrix determinant
JNIEXPORT jdouble JNICALL 
Java_core_utils_MatrixOperations_determinant(JNIEnv* env, jobject, 
    jdoubleArray matrix, jint rows, jint cols) {
    try {
        Matrix a = jArrayToMatrix(env, matrix, rows, cols);
        return a.determinant();
    } catch (const std::exception& e) {
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return 0.0;
    }
}

// matrix inverse
JNIEXPORT jdoubleArray JNICALL 
Java_core_utils_MatrixOperations_inverse(JNIEnv* env, jobject, 
    jdoubleArray matrix, jint rows, jint cols) {
    try {
        Matrix a = jArrayToMatrix(env, matrix, rows, cols);
        Matrix result = a.inverse();
        return matrixToJArray(env, result);
    } catch (const std::exception& e) {
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return nullptr;
    }
}

// matrix trace
JNIEXPORT jdouble JNICALL 
Java_core_utils_MatrixOperations_trace(JNIEnv* env, jobject, 
    jdoubleArray matrix, jint rows, jint cols) {
    try {
        Matrix a = jArrayToMatrix(env, matrix, rows, cols);
        return a.trace();
    } catch (const std::exception& e) {
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return 0.0;
    }
}

// matrix rank
JNIEXPORT jint JNICALL 
Java_core_utils_MatrixOperations_rank(JNIEnv* env, jobject, 
    jdoubleArray matrix, jint rows, jint cols) {
    try {
        Matrix a = jArrayToMatrix(env, matrix, rows, cols);
        return a.rank();
    } catch (const std::exception& e) {
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return 0;
    }
}

// check if matrix is symmetric
JNIEXPORT jboolean JNICALL 
Java_core_utils_MatrixOperations_isSymmetric(JNIEnv* env, jobject, 
    jdoubleArray matrix, jint rows, jint cols) {
    try {
        Matrix a = jArrayToMatrix(env, matrix, rows, cols);
        return a.isSymmetric();
    } catch (const std::exception& e) {
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return false;
    }
}

}