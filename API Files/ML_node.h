// API created by William Shepelak and Carmen Thom 3/19/2023

/*
    This file containes the class used for each node of the ML network
    ...
*/

#include <vector>
#include <math.h>


class ML_node
{
    public:
        ML_node();
    private:
        double weights;
        double bias;

        double ML_identity_func(double);
        double ML_identity_derv(double);
};











// Activation Functions + Derivitives

double ML_node::ML_identity_func(double input)
{
    /*
        args:
            input <double>: the sum of the weights, inputs, and biases as a double

        Use:
            The identity activation function
    
    */
   return input;

}

double ML_node::ML_identity_derv(double input)
{
    /*
        args:
            input <double>: Error Terms

        Use:
            The identity activation derivitive function
    
    */
    return 1.0;
}


