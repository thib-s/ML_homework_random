package shared;

/**
 * An abstract errors measure
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public abstract class AbstractErrorMeasure implements ErrorMeasure {
    /**
     * Calculate the errors between two data sets
     * @param a the first
     * @param b the second
     * @return the errors
     */
    public double value(DataSet a, DataSet b) {
        double error = 0;
        for (int i = 0; i < a.size(); i++) {
            error += value(a.get(i), b.get(i));
        }
        return error;
    }
}
