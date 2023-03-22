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

    private:
        ML_eType network_error;
        std::vector<ML_layer> layer_layer;
        int input_size;
};



ML_network::ML_network(int inputs, ML_eType error_type)
{
    this->input_size = inputs;
    this->network_error = error_type;
}


std::vector<double> ML_network::solve(std::vector<double> x_data)
{
    return {};
}

void ML_network::train(std::vector<std::vector<double>> x_data, std::vector<std::vector<double>> y_data, int epochs, bool trace)
{
    return;
}