import org.jblas.DoubleMatrix;

public class MNISTPair {
    public int label;
    public DoubleMatrix image;
    public MNISTPair(int label, DoubleMatrix image) {
        this.label = label;
        this.image = image;
    }
}
