// API created by William Shepelak and Carmen Thom 3/19/2023

/*
    This file containes the class used for each node of the ML network
    ...
*/

#include <vector>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

enum ML_nType {
    NON_Declared,
    Identity,
    Binary_step,
    Sigmoid,
    Tanh,

    Softplus,
    Leaky_ReLU,
    SiLU,
    Gaussian
};

class ML_node
{
    public:
        ML_node(ML_nType, int);
        double solve_node(std::vector<double>);

    private:
        std::vector<double> weights;
        double bias;
        ML_nType activation_type;
        double get_activation(double);
        double ML_Identity_func(double);
        double ML_Identity_derv(double);
        double ML_Binary_step_func(double);
        double ML_Binary_step_derv(double);
        double ML_Sigmoid_func(double);
        double ML_Sigmoid_derv(double);
        double ML_Tanh_func(double);
        double ML_Tanh_derv(double);

        double ML_Softplus_func(double);
        double ML_Softplus_derv(double);
        double ML_Leaky_ReLU_func(double);
        double ML_Leaky_ReLU_derv(double);
        double ML_PReLU_func(double);
        double ML_PReLU_derv(double);
        double ML_SiLU_func(double);
        double ML_SiLU_derv(double);
        double ML_Gaussian_func(double);
        double ML_Gaussian_derv(double);
};

/// @brief Node constructor used for creating a network from scratch
/// @param activation_type an enumerative type defining which activation type the node is
/// @param weight_size An integer denoting how many inputs are coming from the previous layer
ML_node::ML_node(ML_nType activation_type, int weight_size)
{

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


/// @brief Gets the output of a node
/// @param inputs output doubles from last layer
/// @return A single double coorsponding to a nodes output
double ML_node::solve_node(std::vector<double> inputs)
{
    if(inputs.size() != this->weights.size())
    {
        throw std::invalid_argument("input and weight vector sizes do not match"); //flag

    }

    double temp = 0;
    for(int i=0;i<inputs.size();i++)
    {
        temp += inputs[i]*weights[i];
    }
    temp += this->bias;
    return get_activation(temp);
    
}

double ML_node::get_activation(double x)
{
    switch (this->activation_type)
    {
    case NON_Declared:
        throw std::invalid_argument("faulty activation type");
        break;
    case Identity:
        return ML_Identity_func(x);
        break;
    case Binary_step:
        return ML_Binary_step_func(x);
        break;
    case Sigmoid:
        return ML_Sigmoid_func(x);
        break;
    case Tanh:
        return ML_Tanh_func(x);
        break;
    case Softplus:
        return ML_Softplus_func(x);
        break;
    case Leaky_ReLU:
        return ML_Leaky_ReLU_func(x);
        break;
    case SiLU:
        return ML_SiLU_func(x);
        break;
    case Gaussian:
        return ML_Gaussian_func(x);
        return;
    default:
        throw std::invalid_argument("faulty activation type");
        break;
    }
}


// Activation Functions + Derivitives

/// @brief The identity activation function
/// @param input the sum of the weights, inputs, and biases as a double
/// @return The evaluation of the indentity function w.r.t its inputs
double ML_node::ML_Identity_func(double input)
{
   return input;
}

/// @brief The identity activation derivitive function
/// @param input Error Terms
/// @return The evaluation of the identity activation derivitive function
double ML_node::ML_Identity_derv(double input)
{
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

double ML_Tanh_func(double input)
{
    return (exp(input)-exp(input*-1))/(exp(input)+ exp(input*-1));
}

double ML_Tanh_derv(double input)
{
    return 1 - pow(ML_Tanh_func(input), 2);
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

/// @brief The Gaussian activation function
/// @param inputs input <double>: the sum of the weights, inputs, and biases as a double
/// @return The evaluation of the gaussian function w.r.t its inputs
double ML_node::ML_Gaussian_func(double input)
{
   return exp(pow(input, 2) * -1);

}

/// @brief The Gaussian activation derivitive function
/// @param inputs input <double>: the sum of the weights, inputs, and biases as a double
/// @return The derivative of the gaussian function w.r.t its inputs
double ML_node::ML_Gaussian_derv(double input)
{
    return -2*input*exp(pow(input, 2) * -1);
}