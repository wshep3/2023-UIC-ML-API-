// API created by William Shepelak and Carmen Thom 3/19/2023

#include <vector>
#include <math.h>
#include "ML_node.h"



class ML_layer {

    public:
        ML_layer(ML_nType, int, int);


    private:
        std::vector<ML_node> node_layer;
        ML_nType layer_type;
        std::vector<double> Solve_layer(std::vector<double>);

};



ML_layer::ML_layer(ML_nType ntype, int layer_size, int size_previous)
{
    for(int i=0;i<layer_size;i++)
    {
        ML_node temp = ML_node(ntype, size_previous);
        this->node_layer.push_back(temp);
    }
    this->layer_type = ntype;
}

std::vector<double> ML_layer::Solve_layer(std::vector<double> inputs){
    std::vector<double> outputs;
    for (int i = 0; i < node_layer.size(); i++){
        outputs.push_back(node_layer[i].solve_node(inputs));
    }
    return outputs;
}