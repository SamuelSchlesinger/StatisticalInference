/*

Based on code found at
https://code.google.com/p/pen-ui/source/browse/trunk/skrui/src/org/six11/skrui/charrec/MNISTReader.java?r=185

*/

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import org.jblas.DoubleMatrix;

public class MNISTReader {
    
    private DataInputStream labels;
    private DataInputStream images;
    private int numLabels, numImages, numRows, numCols, numLabelsRead, numImagesRead, length;

    /**
    * Constructs a new MNISTReader with the given filepaths
    */
    public MNISTReader(String labels_fp, String data_fp) throws IOException {
        this.labels = new DataInputStream(new FileInputStream(labels_fp));
        this.images = new DataInputStream(new FileInputStream(data_fp));
        int magicNumber = labels.readInt();
        if (magicNumber != 2049) {
            System.err.printf("Label file has wrong magic number: %d (should be 2049");
            System.exit(0);
        }
        magicNumber = images.readInt();
        if (magicNumber != 2051) {
            System.err.printf("Label file has wrong magic number: %d (should be 2051)");
            System.exit(0);
        }
        this.numLabels = labels.readInt();
        this.numImages = images.readInt();
        this.numRows = images.readInt();
        this.numCols = images.readInt();
        this.length = numRows * numCols;
        if (numLabels != numImages) {
            System.err.println("Image file and label file do not contain the same number of entries.");
            System.err.printf("Label file contains: %d", numLabels);
            System.err.printf("Image file contains: %d", numImages);
            System.exit(0);
        }
        this.numLabelsRead = 0;
        this.numImagesRead = 0;
    }

    /**
    * Returns a row major, binary version of the MNIST data set
    */
    public MNISTPair next() throws IOException {
        DoubleMatrix image = new DoubleMatrix(numRows * numCols);
        for (int i = 0; i < length; i++) {
            image.put(i, images.readUnsignedByte() > 50 ? 1 : 0);
        }
        numImagesRead++;
        int label = labels.readByte();
        numLabelsRead++;
        return new MNISTPair(label, image);
    }

    public boolean hasNext() {
        return numImagesRead < numImages;
    }

    public int numRows() {
        return numRows;
    }

    public int numCols() {
        return numCols;
    }

    public static void printDigit(DoubleMatrix m, int numCols, int numRows) { 
        for (int i = 0; i < numCols; i++) {
            for (int j = 0; j < numRows; j++) {
                System.out.printf("%c", m.get(i * numCols + j) == 1 ? ' ' : '#');
            } System.out.println();
        }
    }

    public static void main(String[] args) {
    }
}
