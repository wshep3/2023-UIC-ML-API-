// API created by William Shepelak and Carmen Thom 3/19/2023

#include <vector>
#include <math.h>
#include "ML_node.h"



class ML_layer {

    public:
        ML_layer(ML_nType activation_type, int layer_size);


    private:
        std::vector<ML_layer> node_layer;


};



ML_layer::ML_layer(ML_nType ntype, int size)
{
    
}