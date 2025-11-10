package core.utils;

public class Matrix {

    static {
        try {
            System.loadLibrary("matrix_jni");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native library failed to load: " + e.getMessage());
            System.exit(1);
        }
    }

    private native double[] addMatrices(double[] a, int ra, int ca, double[] b, int rb, int cb);

    private native double[] subtractMatrices(double[] a, int ra, int ca, double[] b, int rb, int cb);

    private native double[] multiplyMatrices(double[] a, int ra, int ca, double[] b, int rb, int cb);

    private native double[] scalarMultiply(double[] a, int r, int c, double s);

    private native double[] transpose(double[] a, int r, int c);

    private static void printMatrix(double[] m, int r, int c) {
        if (m == null) {
            System.out.println("Matrix is null (operation failed).");
            return;
        }
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                System.out.print(m[i * c + j] + " ");
            }
            System.out.println();
        }
    }
}
