import java.util.Random;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
* A Restricted Boltzmann Machine trained using CD(k)
* as invented by Hinton
*/ 
public class RBM {

    /* Number of visible and hidden neurons */
    private int V, H;

    /* Weight matrix between the visible and hidden layer */
    private DoubleMatrix theta;

    /* Meta parameter involved in training */
    private DoubleMatrix momentum; 

    /* Visible prior/bias */
    private DoubleMatrix visibleBias;

    /* Visible state */
    private DoubleMatrix visible;

    /* Hidden prior/bias */
    private DoubleMatrix hiddenBias;

    /* Hidden state */
    private DoubleMatrix hidden;

    /**
    * Constructor which taken numbers of visible and hidden
    * neurons and constructs a new network
    */
    public RBM(int visible, int hidden) {
        this.V = visible;
        this.H = hidden;
        this.theta = DoubleMatrix.randn(visible, hidden);
        this.momentum = theta.dup();
        this.visibleBias = DoubleMatrix.randn(visible);
        this.visible = DoubleMatrix.randn(visible);
        this.hiddenBias = DoubleMatrix.randn(hidden);
        this.hidden = DoubleMatrix.randn(hidden);
    }

    /**
    * Sets the hidden state
    */
    public void setHidden(DoubleMatrix vec) {
        assert vec.rows == H;
        this.hidden.copy(vec);
    }

    /**
    * Sets the visible state
    */
    public void setVisible(DoubleMatrix vec) {
        assert vec.rows == V;
        this.visible.copy(vec);
    }

    /**
    * Computes the probabilities of whether or not
    * each hidden neuron will be on given the current state of the
    * visible neurons
    */
    public DoubleMatrix hiddenGivenVisible(double beta) {
        DoubleMatrix p = hiddenBias.add(theta.transpose().mmul(visible)).mmuli(-beta);
        MatrixFunctions.expi(p);
        p.addi(1);
        DoubleMatrix P = DoubleMatrix.ones(H).divi(p);
        return P;
    }

    /**
    * Computes the probabilities of whether or not each visible
    * neuron will be on given the current state of the hidden
    * neurons
    */
    public DoubleMatrix visibleGivenHidden(double beta) {
        DoubleMatrix p = visibleBias.add(theta.mmul(hidden)).mmuli(-beta);
        MatrixFunctions.expi(p);
        p.addi(1);
        DoubleMatrix P = DoubleMatrix.ones(V).divi(p);
        return P;
    }

    /**
    * Updates the hidden values stochastically given the visible ones
    */
    public void updateHidden(double beta) {
        DoubleMatrix p = hiddenGivenVisible(beta);
        Random random = new Random();
        for (int i = 0; i < H; i++) {
            if (p.get(i, 0) > random.nextDouble()) {
                hidden.put(i, 0, 1); 
            } else {
                hidden.put(i, 0, 0);
            }
        }
    }

    /**
    * Updates the visible values stochastically given the hidden ones
    */
    public void updateVisible(double beta) {
        DoubleMatrix p = visibleGivenHidden(beta);
        Random random = new Random();
        for (int i = 0; i < V; i++) {
            if (p.get(i, 0) > random.nextDouble()) {
                visible.put(i, 0, 1);
            } else {
                visible.put(i, 0, 0);
            }
        }
    }

    /**
    * Trains the model using CD(k) on the given feature vectors for
    * the given number of iterations
    * Beta is a metaparameter which makes the sigmoid curve steeper 
    * Learning rate is the scaling of the parameter change
    */
    public void CD(int K, DoubleMatrix X, int iterations, double beta, double learningRate) {
        assert X.rows == V;
        Random random = new Random();
        for (int i = 0; i < iterations; i++) {
            int choice = random.nextInt(X.columns);
            DoubleMatrix x = X.getColumn(choice);
            setVisible(x);
            updateHidden(beta);
            DoubleMatrix CD_hidden = hidden.dup();
            DoubleMatrix CD_visible = visible.dup();
            DoubleMatrix CD_positive = hidden.mmul(visible.transpose());
            for (int k = 0; k < K; k++) {
                updateVisible(beta);
                updateHidden(beta);
            }
            hiddenBias.addi(CD_hidden.subi(hidden).mmuli(learningRate));
            visibleBias.addi(CD_visible.subi(visible).mmuli(learningRate));
            DoubleMatrix CD_negative = hidden.mmul(visible.transpose());
            DoubleMatrix changeoooo = CD_positive.subi(CD_negative).mmuli(learningRate);
            theta.addi(changeoooo);
            theta.addi(momentum);
            momentum.addi(changeoooo).mmuli(0.4);
        }
    }

    public static void main(String[] args) {
        RBM rbm = new RBM(10, 20);
        DoubleMatrix X = DoubleMatrix.zeros(10, 5);
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 5; j++) {
                if (i < 4) {
                    X.put(i, j, 1);
                }
            }
        }
        rbm.CD(2, X, 200000, 2, 0.001);
        rbm.setVisible(DoubleMatrix.ones(10));
        rbm.updateHidden(3);
        rbm.updateVisible(5);
        System.out.println(rbm.visible);
    }
}
