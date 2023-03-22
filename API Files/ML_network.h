// API created by William Shepelak and Carmen Thom 3/19/2023

#include <vector>
#include <math.h>
#include "ML_node.h"
#include "ML_layer.h"

enum ML_eType 
{
    NON_Declared,
    MSE,

};


class ML_network
{
    public:
        ML_network(int, ML_eType);
        std::vector<double> solve(std::vector<double>);
        void train(std::vector<std::vector<double>>, std::vector<std::vector<double>>, int, bool);
        void add_layer(int, ML_nType);
        double get_network_error(std::vector<double> x_data, std::vector<double> y_data);

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

double ML_MSE_loss(std::vector<double> solved, std::vector<double> actul){
    double error = 0;
    for (int i = 0; i < solved.size(); i++){
        error += pow(solved[i]-actul[i], 2);
    }
    return (1.0/solved.size())*error;
}