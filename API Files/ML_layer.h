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