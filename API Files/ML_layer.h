// API created by William Shepelak and Carmen Thom 3/19/2023

#include <vector>
#include <math.h>
#include "ML_node.h"



class ML_layer {

    public:
        ML_layer(ML_nType, int, int);
        int get_layer_size();

    private:
        int layer_size;
        std::vector<ML_node> node_layer;
        ML_nType layer_type;
        std::vector<double> Solve_layer(std::vector<double>);

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