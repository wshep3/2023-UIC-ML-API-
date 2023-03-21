// API created by William Shepelak and Carmen Thom 3/19/2023

/*
    This file containes the class used for each node of the ML network
    ...
*/

#include <vector>
#include <math.h>
#include <cstdlib>
#include <ctime>

enum ML_nType {
    NON_Declared,
    identity,
    binary_step,
    sigmoid,

    Softplus,
    Leaky_ReLU,
    SiLU,
    gaussian
};

class ML_node
{
    public:
        ML_node(ML_nType, int);

    private:
        std::vector<double> weights;
        double bias;
        ML_nType activation_type;
        double ML_identity_func(double);
        double ML_identity_derv(double);
        double ML_Binary_step_func(double);
        double ML_Binary_step_derv(double);
        double ML_Sigmoid_func(double);
        double ML_Sigmoid_derv(double);

        double ML_Softplus_func(double);
        double ML_Softplus_derv(double);
        double ML_Leaky_ReLU_func(double);
        double ML_Leaky_ReLU_derv(double);
        double ML_PReLU_func(double);
        double ML_PReLU_derv(double);
        double ML_SiLU_func(double);
        double ML_SiLU_derv(double);
        double ML_gaussian_func(double);
        double ML_gaussian_derv(double);
};

ML_node::ML_node(ML_nType activation_type, int weight_size)
{
    /*
        args:
            activation_type <ML_nType>: a enumerative type defining which activation type the node is
            weight_size <int>: An integer denoting how many inputs are coming from the previous layer

        Use:
            Node constructor used for creating a network from scratch
    */


    /*
        each mode has a bias and weights that are associated with each input
        This looks like a single double which is a bias
        and a vector of weights -> this doesnt have to be dynamic but we are in c++ so who cares
    */

    /*
        Be defult weights should be initiallized with random biases
        because we want an option to load networks I think we will create
        two diffrent constructer methods - this one is for creating a network
        from scratch
    */

   this->activation_type = activation_type;
   std::srand(time(nullptr));
   double rand_num = 0;
   for(int i=0;i<weight_size;i++)
   {
    rand_num = (double)rand() / RAND_MAX;
    rand_num = 2 * rand_num - 1;
    this->weights.push_back(rand_num);
   }
   rand_num = (double)rand() / RAND_MAX;
   rand_num = 2 * rand_num - 1;
   this->bias = rand_num;
}


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
    /*
        args:
            input <double>: the sum of the weights, inputs, and biases as a double
        
        Use:
            The Sigmoid activation function
    */
    return (1.0)/(1.0+exp(-1*input));
}

double ML_node::ML_Sigmoid_derv(double input)
{
    /*
        args:
            input <double>: Error Terms
        
        Use:
            The Sigmoid activation derivitive function
    */
    return ML_Sigmoid_func(input)*(1.0-ML_Sigmoid_func(input));
}

double ML_Softplus_func(double input)
{
    log(1+ exp(input));
}

double ML_Softplus_derv(double input)
{
    return 1/(1+exp(input));
}

double ML_Leaky_ReLU_func(double input)
{
    if(input<0){return 0.01*input;}
    else{return input;}
}
        
double ML_Leaky_ReLU_derv(double input)
{
    if(input<0){return 0.01;}
    else{return 1;}
}

/*
double ML_node::ML_PReLU_func(double input)
   int a;
   if(input>=0){return input;} 
   else{return input*a;}
}

double ML_node::ML_PReLU_derv(double input)
{
   int a;
    if(input<0){return a;}
    else{return 1;}
}

*/

double ML_node::ML_SiLU_func(double input)
{

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