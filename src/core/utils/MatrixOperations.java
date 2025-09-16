package core.utils;

public class MatrixOperations {

    static {
        System.loadLibrary("matrixops");
    }

    // native function declarations
    public native double[] addMatrices(double[] matA, int rowsA, int colsA,
            double[] matB, int rowsB, int colsB);

    public native double[] subtractMatrices(double[] matA, int rowsA, int colsA,
            double[] matB, int rowsB, int colsB);

    public native double[] multiplyMatrices(double[] matA, int rowsA, int colsA,
            double[] matB, int rowsB, int colsB);

    public native double[] scalarMultiply(double[] matrix, int rows, int cols, double scalar);

    public native double[] transpose(double[] matrix, int rows, int cols);

    public native double determinant(double[] matrix, int rows, int cols);

    public native double[] inverse(double[] matrix, int rows, int cols);

    public native double trace(double[] matrix, int rows, int cols);

    public native int rank(double[] matrix, int rows, int cols);

    public native boolean isSymmetric(double[] matrix, int rows, int cols);

    // helper methods for matrix manipulation

    /**
     * convert 2d array to 1d array (row-major order)
     */
    public static double[] flatten(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] result = new double[rows * cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i * cols + j] = matrix[i][j];
            }
        }
        return result;
    }

    /**
     * convert 1d array to 2d array (row-major order)
     */
    public static double[][] unflatten(double[] array, int rows, int cols) {
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = array[i * cols + j];
            }
        }
        return result;
    }

    /**
     * print matrix in readable format
     */
    public static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            for (double val : row) {
                System.out.printf("%8.3f ", val);
            }
            System.out.println();
        }
    }

    /**
     * validate matrix dimensions for operations
     */
    private static void validateDimensions(double[][] matA, double[][] matB, String operation) {
        if (operation.equals("add") || operation.equals("subtract")) {
            if (matA.length != matB.length || matA[0].length != matB[0].length) {
                throw new IllegalArgumentException("matrices must have same dimensions for " + operation);
            }
        } else if (operation.equals("multiply")) {
            if (matA[0].length != matB.length) {
                throw new IllegalArgumentException("invalid dimensions for matrix multiplication");
            }
        }
    }

    /**
     * validate square matrix
     */
    private static void validateSquareMatrix(double[][] matrix, String operation) {
        if (matrix.length != matrix[0].length) {
            throw new IllegalArgumentException(operation + " requires square matrix");
        }
    }

    // high-level matrix operations with 2d arrays

    /**
     * matrix addition with 2d arrays
     */
    public double[][] add(double[][] matA, double[][] matB) {
        validateDimensions(matA, matB, "add");
        double[] flatA = flatten(matA);
        double[] flatB = flatten(matB);
        double[] result = addMatrices(flatA, matA.length, matA[0].length,
                flatB, matB.length, matB[0].length);
        return unflatten(result, matA.length, matA[0].length);
    }

    /**
     * matrix subtraction with 2d arrays
     */
    public double[][] subtract(double[][] matA, double[][] matB) {
        validateDimensions(matA, matB, "subtract");
        double[] flatA = flatten(matA);
        double[] flatB = flatten(matB);
        double[] result = subtractMatrices(flatA, matA.length, matA[0].length,
                flatB, matB.length, matB[0].length);
        return unflatten(result, matA.length, matA[0].length);
    }

    /**
     * matrix multiplication with 2d arrays
     */
    public double[][] multiply(double[][] matA, double[][] matB) {
        validateDimensions(matA, matB, "multiply");
        double[] flatA = flatten(matA);
        double[] flatB = flatten(matB);
        double[] result = multiplyMatrices(flatA, matA.length, matA[0].length,
                flatB, matB.length, matB[0].length);
        return unflatten(result, matA.length, matB[0].length);
    }

    /**
     * scalar multiplication with 2d arrays
     */
    public double[][] multiply(double[][] matrix, double scalar) {
        double[] flat = flatten(matrix);
        double[] result = scalarMultiply(flat, matrix.length, matrix[0].length, scalar);
        return unflatten(result, matrix.length, matrix[0].length);
    }

    /**
     * matrix transpose with 2d arrays
     */
    public double[][] transpose(double[][] matrix) {
        double[] flat = flatten(matrix);
        double[] result = transpose(flat, matrix.length, matrix[0].length);
        return unflatten(result, matrix[0].length, matrix.length);
    }

    /**
     * matrix determinant with 2d arrays
     */
    public double determinant(double[][] matrix) {
        validateSquareMatrix(matrix, "determinant");
        double[] flat = flatten(matrix);
        return determinant(flat, matrix.length, matrix[0].length);
    }

    /**
     * matrix inverse with 2d arrays
     */
    public double[][] inverse(double[][] matrix) {
        validateSquareMatrix(matrix, "inverse");
        double[] flat = flatten(matrix);
        double[] result = inverse(flat, matrix.length, matrix[0].length);
        return unflatten(result, matrix.length, matrix[0].length);
    }

    /**
     * matrix trace with 2d arrays
     */
    public double trace(double[][] matrix) {
        validateSquareMatrix(matrix, "trace");
        double[] flat = flatten(matrix);
        return trace(flat, matrix.length, matrix[0].length);
    }

    /**
     * matrix rank with 2d arrays
     */
    public int rank(double[][] matrix) {
        double[] flat = flatten(matrix);
        return rank(flat, matrix.length, matrix[0].length);
    }

    /**
     * check if matrix is symmetric with 2d arrays
     */
    public boolean isSymmetric(double[][] matrix) {
        if (matrix.length != matrix[0].length) return false;
        double[] flat = flatten(matrix);
        return isSymmetric(flat, matrix.length, matrix[0].length);
    }

    /**
     * create identity matrix
     */
    public static double[][] identity(int size) {
        double[][] result = new double[size][size];
        for (int i = 0; i < size; i++) {
            result[i][i] = 1.0;
        }
        return result;
    }

    /**
     * create zero matrix
     */
    public static double[][] zeros(int rows, int cols) {
        return new double[rows][cols];
    }

    /**
     * create matrix filled with ones
     */
    public static double[][] ones(int rows, int cols) {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = 1.0;
            }
        }
        return result;
    }

    /**
     * create random matrix with values between 0 and 1
     */
    public static double[][] random(int rows, int cols) {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.random();
            }
        }
        return result;
    }

    /**
     * copy matrix
     */
    public static double[][] copy(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(matrix[i], 0, result[i], 0, cols);
        }
        return result;
    }

    /**
     * matrix equality check with tolerance
     */
    public static boolean equals(double[][] matA, double[][] matB, double tolerance) {
        if (matA.length != matB.length || matA[0].length != matB[0].length) {
            return false;
        }
        
        for (int i = 0; i < matA.length; i++) {
            for (int j = 0; j < matA[0].length; j++) {
                if (Math.abs(matA[i][j] - matB[i][j]) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * matrix equality check with default tolerance
     */
    public static boolean equals(double[][] matA, double[][] matB) {
        return equals(matA, matB, 1e-10);
    }

    // example usage and testing
    public static void main(String[] args) {
        MatrixOperations ops = new MatrixOperations();

        // create test matrices
        double[][] matA = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };

        double[][] matB = {
            {9.0, 8.0, 7.0},
            {6.0, 5.0, 4.0},
            {3.0, 2.0, 1.0}
        };

        System.out.println("matrix a:");
        printMatrix(matA);

        System.out.println("\nmatrix b:");
        printMatrix(matB);

        // test addition
        System.out.println("\na + b:");
        printMatrix(ops.add(matA, matB));

        // test subtraction
        System.out.println("\na - b:");
        printMatrix(ops.subtract(matA, matB));

        // test multiplication
        System.out.println("\na * b:");
        printMatrix(ops.multiply(matA, matB));

        // test scalar multiplication
        System.out.println("\na * 2.0:");
        printMatrix(ops.multiply(matA, 2.0));

        // test transpose
        System.out.println("\ntranspose of a:");
        printMatrix(ops.transpose(matA));

        // test determinant
        System.out.println("\ndeterminant of a: " + ops.determinant(matA));

        // test rank
        System.out.println("rank of a: " + ops.rank(matA));

        // test trace
        System.out.println("trace of a: " + ops.trace(matA));

        // test symmetry
        System.out.println("is a symmetric: " + ops.isSymmetric(matA));

        // test with invertible matrix
        double[][] invertible = {
            {4.0, 7.0},
            {2.0, 6.0}
        };

        System.out.println("\ninvertible matrix:");
        printMatrix(invertible);

        System.out.println("\ninverse:");
        double[][] inv = ops.inverse(invertible);
        printMatrix(inv);

        System.out.println("\nverification (should be identity):");
        printMatrix(ops.multiply(invertible, inv));

        System.out.println("\ndeterminant: " + ops.determinant(invertible));
        System.out.println("trace: " + ops.trace(invertible));
        System.out.println("is symmetric: " + ops.isSymmetric(invertible));

        // test utility functions
        System.out.println("\n3x3 identity matrix:");
        printMatrix(identity(3));

        System.out.println("\n2x3 zeros matrix:");
        printMatrix(zeros(2, 3));

        System.out.println("\n2x2 ones matrix:");
        printMatrix(ones(2, 2));

        // test matrix equality
        double[][] mat1 = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] mat2 = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] mat3 = {{1.0, 2.0}, {3.0, 4.001}};
        
        System.out.println("\nequality test (exact): " + equals(mat1, mat2));
        System.out.println("equality test (tolerance): " + equals(mat1, mat3, 0.01));
        System.out.println("equality test (strict): " + equals(mat1, mat3));
    }
}