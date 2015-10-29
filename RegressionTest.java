import java.util.Random;
import org.jblas.DoubleMatrix;

/**
* This piece of code is meant to test the LogisticRegression
* as well as LinearRegression classes
*/
public class RegressionTest {

    private static Random random = new Random();

    private static void testLinear(int L, int N) {
        DoubleMatrix fn = DoubleMatrix.rand(L);
        double bias = random.nextDouble();
        DoubleMatrix X = DoubleMatrix.rand(N, L);
        DoubleMatrix Y = new DoubleMatrix(N, 1);
        for (int i = 0; i < N; i++) {
            Y.put(i, 0, X.getRow(i).dot(fn) + bias);   
        }
        LinearRegression regression = new LinearRegression(X, Y, 0.01, 1000000);
        DoubleMatrix testX = DoubleMatrix.rand(10, L);
        double error = 0;
        for (int i = 0; i < 10; i++) {
            error += Math.abs(regression.predict(testX.getRow(i)) - (testX.getRow(i).dot(fn) + bias));
        }
        System.out.printf("LinearRegression total_error = %f, avg_error = %f\n", error, error/10);
    }

    private static void testLogistic(int L, int N) {
        DoubleMatrix fn = DoubleMatrix.rand(L);
        double bias = random.nextDouble();
        DoubleMatrix X = DoubleMatrix.rand(N, L);
        DoubleMatrix Y = new DoubleMatrix(N, 1);
        for (int i = 0; i < N; i++) {
            Y.put(i, 0, 1 / (1 + Math.exp(-X.getRow(i).dot(fn) - bias)));   
        }
        LogisticRegression regression = new LogisticRegression(X, Y, 0.01, 1000000);
        DoubleMatrix testX = DoubleMatrix.rand(10, L);
        double error = 0;
        for (int i = 0; i < 10; i++) {
            error += Math.abs(regression.predict(testX.getRow(i)) - (1 / (1 + Math.exp(-testX.getRow(i).dot(fn) - bias))));
        }
        System.out.printf("LogisticRegression total_error = %f, avg_error = %f\n", error, error/10);
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("usage: java Test featurelength numberofsamples");
            System.exit(1);
        }
        int L = Integer.parseInt(args[0]);
        int N = Integer.parseInt(args[1]);
        testLinear(L, N);
        testLogistic(L, N);
        
    }
}
