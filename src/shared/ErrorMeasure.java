package shared;

/**
 * A class representing an errors measure
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public interface ErrorMeasure {

    /**
     * Measure the errors for the given output and target
     * @param output the output
     * @param example the example
     * @return the errors
     */
    public abstract double value(Instance output, Instance example);

}
