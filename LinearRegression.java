import java.util.Random;
import org.jblas.DoubleMatrix;

/**
* Multivarirate linear regression model which fits given feature
* vectors to a real value
*/
public class LinearRegression {

    /* Learned vector */
    private DoubleMatrix theta;

    /* Length of input */
    private int L;

    /* Intercept value */
    private double bias;

    public LinearRegression(DoubleMatrix X, DoubleMatrix Y, double learningRate, int iterations) {
        assert X.rows == Y.rows;
        this.L = X.columns;
        this.theta = DoubleMatrix.ones(X.columns);
        this.bias = 0;
        Random random = new Random();
        for (int i = 0; i < iterations; i++) {
            int choice = random.nextInt(X.rows);
            DoubleMatrix x = X.getRow(choice);
            double prediction = predict(x);
            double target = Y.get(choice, 0);
            double error = prediction - target; 
            theta.subi(X.getRow(choice).mmul(learningRate*error)); // alter theta by gradient descent
            bias -= error * learningRate;
        }
    }

    public double predict(DoubleMatrix x) {
        assert x.columns == L;
        return theta.dot(x) + bias;
    }
}
