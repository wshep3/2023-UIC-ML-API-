// API created by William Shepelak and Carmen Thom 3/19/2023

#include <vector>
#include <math.h>
#include "ML_node.h"
#include "ML_layer.h"
#pragma once

enum ML_eType 
{
    NON,
    MSE,

};


class ML_network
{
    public:
        ML_network(int, ML_eType);
        std::vector<double> solve(std::vector<double>);
        void train(std::vector<std::vector<double>>, std::vector<std::vector<double>>, int, bool);
        void add_layer(int, ML_nType);
        double get_network_error(std::vector<double>, std::vector<double>);

    private:
        double ML_MSE_loss(std::vector<double> solved, std::vector<double> actul);
        ML_eType network_error;
        std::vector<ML_layer> network_vector;
        int input_size;
};



ML_network::ML_network(int inputs, ML_eType error_type)
{
    this->input_size = inputs;
    this->network_error = error_type;
}

/// @brief Adds a layer to the network
/// @param nodes Data of the nodes
/// @param activation_type Activation function of the node
void ML_network::add_layer(int nodes, ML_nType activation_type)
{
    if(network_vector.size() == 0)
    {
        ML_layer temp_layer = ML_layer(activation_type, nodes, this->input_size);
        network_vector.push_back(temp_layer);
        return;
    }
    ML_layer temp_layer = ML_layer(activation_type, nodes, network_vector[network_vector.size()-1].get_layer_size());
    network_vector.push_back(temp_layer);
}


/// @brief Solves the layers of the network
/// @param x_data Input layer
/// @return Solved network output
std::vector<double> ML_network::solve(std::vector<double> x_data)
{
    for(int i = 0; i < network_vector.size(); i++){
        x_data = network_vector[i].Solve_layer(x_data);
    }
    return x_data;
}

void ML_network::train(std::vector<std::vector<double>> x_data, std::vector<std::vector<double>> y_data, int epochs, bool trace)
{
    return;
}


/// @brief Calls the respective cost function of the network
/// @param x_data Double vector of the output layer
/// @param y_data Double vector of input values
/// @return Output of cost function
double ML_network::get_network_error(std::vector<double> x_data, std::vector<double> y_data)
{
    std::vector<double> y_preds = solve(x_data);
    switch (this->network_error)
    {
    case NON:
        throw std::invalid_argument("faulty error type");
        break;
    case MSE:
        return ML_MSE_loss(y_preds, y_data);
    default:
        throw std::invalid_argument("faulty error type");
        break;
    }
}

/// @brief Mean squared error cost function
/// @param solved Double vector of the output layer
/// @param actul Double vector of input values
/// @return Mean squared error
double ML_network::ML_MSE_loss(std::vector<double> solved, std::vector<double> actul){
    double error = 0;
    for (int i = 0; i < solved.size(); i++){
        error += pow(solved[i]-actul[i], 2);
    }
    return (1.0/solved.size())*error;
}