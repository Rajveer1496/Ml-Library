#include <iostream>
#include <cstdlib>
#include <jni.h>
#include "core_utils_Matrix.h"

using namespace std;

class Matrix {
    int rows, cols;
    double** data;

public:
    Matrix() : rows(0), cols(0), data(nullptr) {}

    Matrix(int r, int c) : rows(r), cols(c) {
        if (r <= 0 || c <= 0)
            throw "row or column <= 0\ndynamic memory allocation failed";
        data = (double**)malloc(rows * sizeof(double*));
        for (int i = 0; i < rows; i++) {
            data[i] = (double*)calloc(cols, sizeof(double));
        }
    }

    Matrix(const Matrix& o) : rows(o.rows), cols(o.cols) {
        if (rows <= 0 || cols <= 0)
            throw "row or col <= 0\ndynamic memory allocation failed";
        data = (double**)malloc(rows * sizeof(double*));
        for (int i = 0; i < rows; i++) {
            data[i] = (double*)malloc(cols * sizeof(double));
            for (int j = 0; j < cols; j++) data[i][j] = o.data[i][j];
        }
    }

    ~Matrix() {
        if (data) {
            for (int i = 0; i < rows; i++) free(data[i]);
            free(data);
        }
    }

    int r() const { return rows; }
    int c() const { return cols; }

    double& at(int i, int j) {
        if (i < 0 || i >= rows || j < 0 || j >= cols)
            throw "i < 0 or i >= rows or j < 0 or j >= cols\ngetter failed";
        return data[i][j];
    }

    double at(int i, int j) const {
        if (i < 0 || i >= rows || j < 0 || j >= cols)
            throw "i < 0 or i >= rows or j < 0 or j >= cols\ngetter failed";
        return data[i][j];
    }

    Matrix operator+(const Matrix& b) const {
        if (rows != b.rows || cols != b.cols)
            throw "rows(A) != rows(B) or cols(A) != cols(B)\naddition operator failed";
        Matrix r(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r.data[i][j] = data[i][j] + b.data[i][j];
        return r;
    }

    Matrix operator-(const Matrix& b) const {
        if (rows != b.rows || cols != b.cols)
            throw "rows(A) != rows(B) or cols(A) != cols(B)\nsubtraction operator failed";
        Matrix r(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r.data[i][j] = data[i][j] - b.data[i][j];
        return r;
    }

    Matrix operator*(const Matrix& b) const {
        int r1 = rows, c1 = cols, r2 = b.rows, c2 = b.cols;
        if (c1 != r2) throw "cols(A) != rows(B)\nvector multiplication failed";
        Matrix result(r1, c2);
        for (int i = 0; i < r1; i++)
            for (int j = 0; j < c2; j++)
                for (int k = 0; k < c1; k++)
                    result.data[i][j] += data[i][k] * b.data[k][j];
        return result;
    }

    Matrix operator*(double s) const {
        Matrix r(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r.data[i][j] = data[i][j] * s;
        return r;
    }

    Matrix transpose() const {
        Matrix r(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r.data[j][i] = data[i][j];
        return r;
    }
};

Matrix javaArray(JNIEnv* env, jdoubleArray arr, jint rows, jint cols) {
    jdouble* e = env->GetDoubleArrayElements(arr, nullptr);
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.at(i, j) = e[i * cols + j];
    env->ReleaseDoubleArrayElements(arr, e, JNI_ABORT);
    return m;
}

jdoubleArray javaArray(JNIEnv* env, const Matrix& m) {
    int size = m.r() * m.c();
    jdoubleArray arr = env->NewDoubleArray(size);
    jdouble* flat = (jdouble*)malloc(size * sizeof(jdouble));
    int idx = 0;
    for (int i = 0; i < m.r(); i++)
        for (int j = 0; j < m.c(); j++)
            flat[idx++] = m.at(i, j);
    env->SetDoubleArrayRegion(arr, 0, size, flat);
    free(flat);
    return arr;
}

extern "C" {

JNIEXPORT jdoubleArray JNICALL Java_core_utils_Matrix_addMatrices
(JNIEnv* env, jobject, jdoubleArray a, jint ra, jint ca, jdoubleArray b, jint rb, jint cb) {
    try {
        Matrix A = javaArray(env, a, ra, ca);
        Matrix B = javaArray(env, b, rb, cb);
        return javaArray(env, A + B);
    } catch (const char* err) {
        cerr << err << endl; cerr.flush();
        return nullptr;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_core_utils_Matrix_subtractMatrices
(JNIEnv* env, jobject, jdoubleArray a, jint ra, jint ca, jdoubleArray b, jint rb, jint cb) {
    try {
        Matrix A = javaArray(env, a, ra, ca);
        Matrix B = javaArray(env, b, rb, cb);
        return javaArray(env, A - B);
    } catch (const char* err) {
        cerr << err << endl; cerr.flush();
        return nullptr;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_core_utils_Matrix_multiplyMatrices
(JNIEnv* env, jobject, jdoubleArray a, jint ra, jint ca, jdoubleArray b, jint rb, jint cb) {
    try {
        Matrix A = javaArray(env, a, ra, ca);
        Matrix B = javaArray(env, b, rb, cb);
        return javaArray(env, A * B);
    } catch (const char* err) {
        cerr << err << endl; cerr.flush();
        return nullptr;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_core_utils_Matrix_scalarMultiply
(JNIEnv* env, jobject, jdoubleArray a, jint r, jint c, jdouble s) {
    try {
        Matrix A = javaArray(env, a, r, c);
        return javaArray(env, A * s);
    } catch (const char* err) {
        cerr << err << endl; cerr.flush();
        return nullptr;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_core_utils_Matrix_transpose
(JNIEnv* env, jobject, jdoubleArray a, jint r, jint c) {
    try {
        Matrix A = javaArray(env, a, r, c);
        return javaArray(env, A.transpose());
    } catch (const char* err) {
        cerr << err << endl; cerr.flush();
        return nullptr;
    }
}

}
