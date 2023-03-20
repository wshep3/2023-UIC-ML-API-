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
        double ML_Binary_step_func(double);
        double ML_Binary_step_derv(double);
        double ML_Sigmoid_func(double);
        double ML_Sigmoid_derv(double);

        double ML_SiLU_func(double);
        double ML_SiLU_derv(double);
        double ML_gaussian_func(double);
        double ML_gaussian_derv(double);
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

double ML_node::ML_Binary_step_func(double input)
{
    /*
        args:
            input <double>: the sum of the weights, inputs, and biases as a double
        
        Use:
            The Binary step activation function
    */
    if(input >= 0){return 1;}
    else{return 0;}
}

double ML_node::ML_Binary_step_derv(double input)
{
    /*
        args:
            input <double>: Error Terms
        
        Use:
            The Binary step activation derivitive function
    */
    return 0;
}

double ML_node::ML_Sigmoid_func(double input)
{
    return (1.0)/(1.0+exp(-1*input));
}

double ML_node::ML_Sigmoid_derv(double input)
{
    return ML_Sigmoid_func(input)*(1.0-ML_Sigmoid_func(input));
}

double ML_node::ML_SiLU_func(double input)
{
    /*
        args:
            input <double>: the sum of the weights, inputs, and biases as a double

        Use:
            The identity activation function
    
    */
   return input / (1+exp(-input));
}

double ML_node::ML_SiLU_derv(double input)
{
    /*
        args:
            input <double>: Error Terms

        Use:
            The identity activation derivitive function
    
    */
    return ((1+exp(-input))+(input*exp(-input)))/ pow(1+exp(-input),2);
}


double ML_node::ML_gaussian_func(double input)
{
    /*
        args:
            input <double>: the sum of the weights, inputs, and biases as a double

        Use:
            The identity activation function
    
    */
   return exp(pow(input, 2) * -1);

}

double ML_node::ML_gaussian_derv(double input)
{
    /*
        args:
            input <double>: Error Terms

        Use:
            The identity activation derivitive function
    
    */
    return -2*input*exp(pow(input, 2) * -1);
}