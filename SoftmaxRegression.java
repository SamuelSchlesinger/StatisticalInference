import java.util.Random;
import org.jblas.*;

/**
* Softmax or Multinomial Regression
*/
public class SoftmaxRegression {

    /* Number of features in each input vector */
    private int L;
    
    /* Number of classes */
    private int K;

    /* L x K trained matrix */
    private DoubleMatrix theta;

    /**
    * Constructor which takes an L x N matrix X,
    * a K x N matrix Y in {0, 1}^N, and fits theta
    * to classify vectors based on these training examples
    */
    public SoftmaxRegression(DoubleMatrix X, DoubleMatrix Y, double learningRate, int iterations) {
        assert X.columns == Y.columns;
        this.L = X.rows;
        this.K = Y.rows;
        int N = X.columns;
        this.theta = DoubleMatrix.randn(L, K);
        Random random = new Random();
        for (int i = 0; i < iterations; i++) {
            int choice = random.nextInt(N);
            DoubleMatrix x = X.getColumn(choice);
            DoubleMatrix y = Y.getColumn(choice);
            y.subi(_probabilities(theta, x));
            theta.addi(x.mmul(y.transpose()).mmuli(learningRate));
        }
    }

    public DoubleMatrix probabilities(DoubleMatrix x) {
        return _probabilities(theta, x);
    }

    private static DoubleMatrix _probabilities(DoubleMatrix theta, DoubleMatrix x) {
        assert theta.columns == x.rows;
        DoubleMatrix P = theta.transpose().mmul(x);
        MatrixFunctions.expi(P);
        double sum = P.sum();
        P.divi(sum);
        return P;
    }

    public static void main(String[] args) {
        int L = Integer.parseInt(args[0]);
        int K = Integer.parseInt(args[1]);
        int N = Integer.parseInt(args[2]);
        DoubleMatrix theta = DoubleMatrix.rand(L, K);
        DoubleMatrix X = DoubleMatrix.rand(L, N);
        DoubleMatrix Y = DoubleMatrix.zeros(K, N);
        DoubleMatrix Y_probabilities = new DoubleMatrix(K, N);
        for (int i = 0 ; i < N; i++) {
            DoubleMatrix yprob = _probabilities(theta, X.getColumn(i));
            Y_probabilities.putColumn(i, yprob);
            int argmax = yprob.argmax();
            Y.put(argmax, i, 1);
        }
        SoftmaxRegression softmax = new SoftmaxRegression(X, Y, 0.0001, 1000000); 
        double error = 0;
        int n_wrong = 0;
        for (int i = 0; i < N; i++) {
            DoubleMatrix x = DoubleMatrix.rand(L, 1);
            DoubleMatrix yprob = _probabilities(theta, x);
            int tk = yprob.argmax();
            DoubleMatrix y = softmax.probabilities(x);
            int yk = y.argmax();
            if (tk != yk) n_wrong += 1; 
            error += MatrixFunctions.absi(yprob.subi(softmax.probabilities(x))).sum();
        }
        System.out.printf("sum_error = %f, avg_error = %f, n_wrong = %d / %d\n", error, error/N, n_wrong, N);
    }
}
