import java.util.Random;
import org.jblas.DoubleMatrix;

/**
* A multivariate, single class logistic regression
* Given a data matrix and a vector in {0, 1}, it trains a classification
* vector using gradient descent to classify possibly unseen feature vectors.
*/
public class LogisticRegression {

    /* Vector which is trained */
    private DoubleMatrix theta; 

    /* Length of feature vector */
    private int L;

    /* Bias value, can be thought of as a prior or an intercept */
    private double bias;
    
    /**
    * Constructor which, given a data matrix and a vector in {0, 1}, will
    * learn a model which will hopefully be able to classify future examples
    */
    public LogisticRegression(DoubleMatrix X, DoubleMatrix Y, double learningRate, int iterations) {
        assert X.rows == Y.rows;
        this.theta = DoubleMatrix.zeros(X.columns);
        this.L = X.columns;
        this.bias = 0;
        Random random = new Random();
        for (int i = 0; i < iterations; i++) {
            int choice = random.nextInt(X.rows); // choose a random data point
            DoubleMatrix x = X.getRow(choice); 
            double p_y = predict(x); 
            double y = Y.get(choice, 0);
            theta.subi(x.mmul(p_y - y)); // update theta using gradient descent
            bias -= p_y - y; // same for the bias
        }
    }

    /**
    * Returns the probability that the given feature vector is a positive instance
    */
    public double predict(DoubleMatrix x) {
        assert x.rows == L;
        return 1 / (1 + Math.exp( - theta.dot(x) - bias));
    }

    /**
    * Given a feature vector and a threshold probability, returns true if the probability
    * of the given vector being a positive instance is greater than the threshold, false otherwise
    */
    public boolean classify(DoubleMatrix x, double threshold) {
        assert x.rows == L;
        return predict(x) > threshold; 
    }
}
