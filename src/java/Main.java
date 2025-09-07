//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
//ignore errors, it is working, let it work
import core.utils.matrix.MatrixOperations;

class devdi{
    public String toString(){

        return null;
    }
}
public class Main {
    public static void main(String[] args) {
        //TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
        // to see how IntelliJ IDEA suggests fixing it.
        System.out.printf("Hello and welcome!");

        for (int i = 1; i <= 5; i++) {
            //TIP Press <shortcut actionId="Debug"/> to start debugging your code. We have set one <icon src="AllIcons.Debugger.Db_set_breakpoint"/> breakpoint
            // for you, but you can always add more by pressing <shortcut actionId="ToggleLineBreakpoint"/>.
            System.out.println("i = " + i);
        }

        //Testing Matrix Operations
        MatrixOperations ops = new MatrixOperations();

        double[][] matA = {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 },
                { 7.0, 8.0, 9.0 }
        };

        double[][] matB = {
                { 9.0, 8.0, 7.0 },
                { 6.0, 5.0, 4.0 },
                { 3.0, 2.0, 1.0 }
        };

        System.out.println("Matrix A:");
        ops.printMatrix(matA);

        System.out.println("\nMatrix B:");
        ops.printMatrix(matB);

        // Test addition
        System.out.println("\nA + B:");
        ops.printMatrix(ops.add(matA, matB));
    }
}