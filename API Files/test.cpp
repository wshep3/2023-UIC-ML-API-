#include "ML_network.h"
#include <iostream>
#include <vector>



int main()
{
    ML_network my_network = ML_network(2, MSE);
    my_network.add_layer(2, Tanh);
    my_network.add_layer(4, Tanh);
    my_network.add_layer(2, Tanh);
    my_network.add_layer(1, Tanh);
    std::cout << my_network.solve({0,1})[0] << std::endl;
    std::cout << my_network.get_network_error({0,1},{1}) << std::endl;
    return(0);
}