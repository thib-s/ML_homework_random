package func.nn.backprop;

import func.nn.Link;



/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class BackPropagationLink extends Link {

    /**
     * The derivative of the errors function
     * in respect to this node, or in the case
     * of batch training possibly the sum
     * of derivative of the errors functions for
     * many patterns.
     */
    private double error;
    
    /**
     * The last derivative of the errors function
     * in respect to this node, sometimes
     * used in training algorithms that use
     * momentum type terms.
     */
    private double lastError;
    
    /**
     * The last change made to this link (last delta),
     * sometimes used in algorithms with momentum
     * type terms.
     */
    private double lastChange;
    
    /**
     * A learning rate term which is used in
     * some algorithms that have weight specific
     * learning rates.
     */
    private double learningRate;
    
    /**
     * @see nn.Link#changeWeight(double)
     */
    public void changeWeight(double delta) {
         super.changeWeight(delta);
         lastChange = delta;
    }
    
    /**
     * Backpropagate errors values into this link
     */
    public void backpropagate() {
        addError(getInValue() * getOutError());
    }
    
    /**
     * Add errors to this link
     * @param error the errors to add
     */
    public void addError(double error) {
        this.error += error;
    }
    
    /**
     * Clear out the errors and
     * set the current errors to be the last errors
     */
    public void clearError() {
        lastError = error;
        error = 0;
    }
    
    /**
     * Get the errors derivative with respect to this weight
     * @return the errors derivative value
     */
    public double getError() {
        return error;
    }
    
    /**
     * Set the errors
     * @param error the errors to set
     */
    public void setError(double error) {
    	this.error = error;
    }

    /**
     * Get the last change in the weight
     * @return the last change in weight
     */
    public double getLastChange() {
        return lastChange;
    }

    /**
     * Get the last errors value
     * @return the last errors value
     */
    public double getLastError() {
        return lastError;
    }
    
    /**
     * Set the learning rate
     * @param learningRate the learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Get the learning rate
     * @return the learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * Get the output errors
     * @return the output errors
     */
    public double getOutError() {
        return ((BackPropagationNode) getOutNode()).getInputError();
    }
    
    /**
     * Get the weighted output errors
     * @return the output errors times the weigh tof the link
     */
    public double getWeightedOutError() {
        return ((BackPropagationNode) getOutNode()).getInputError()
            * getWeight();
    }
    
    /**
     * Get the input errors
     * @return the input errors
     */
    public double getInError() {
        return ((BackPropagationNode) getInNode()).getInputError();
    }

    /**
     * Get the weighted input errors
     * @return the weighted errors
     */
    public double getWeightedInError() {
        return ((BackPropagationNode) getInNode()).getInputError()
             * getWeight();
    }
}
