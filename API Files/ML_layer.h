// API created by William Shepelak and Carmen Thom 3/19/2023

#include "ML_node.h"
#pragma once



class ML_layer {

    public:
        ML_layer(ML_nType, int, int);
        int get_layer_size();
        std::vector<double> Solve_layer(std::vector<double>);
        std::string save_layer();

    private:
        int layer_size;
        std::vector<ML_node> node_layer;
        ML_nType layer_type;
        

};



ML_layer::ML_layer(ML_nType ntype, int layer_size, int size_previous)
{
    this->layer_size = layer_size;
    for(int i=0;i<layer_size;i++)
    {
        ML_node temp = ML_node(ntype, size_previous);
        this->node_layer.push_back(temp);
    }
    this->layer_type = ntype;
}


/// @brief Solves all nodes of a layer
/// @param inputs inputs <double>:  the sum of the weights, inputs, and biases as a double
/// @return Outputs a vector of node evaluations
std::vector<double> ML_layer::Solve_layer(std::vector<double> inputs){
    std::vector<double> outputs;
    for (int i = 0; i < node_layer.size(); i++){
        outputs.push_back(node_layer[i].solve_node(inputs));
    }
    return outputs;
}


int ML_layer::get_layer_size()
{
    return this->layer_size;
}

std::string ML_layer::save_layer()
{
    std::string layer_data = "";
    switch (this->layer_type)
    {
    case NON_Declared:
        throw std::invalid_argument("faulty activation type");
        break;
    case Identity:
        layer_data += "Identity:";
        break;
    case Binary_step:
        layer_data += "Binary_step";
        break;
    case Sigmoid:
        layer_data += "Sigmoid";
        break;
    case Tanh:
        layer_data += "Tanh";
        break;
    case Softplus:
        layer_data += "Softplus";
        break;
    case Leaky_ReLU:
        layer_data += "LeakyReLU";
        break;
    case SiLU:
        layer_data += "SiLU";
        break;
    case Gaussian:
        layer_data += "Gaussian";
        break;
    default:
        throw std::invalid_argument("faulty activation type");
        break;
    }
    // more to come!
}