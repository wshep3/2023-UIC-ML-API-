// API created by William Shepelak and Carmen Thom 3/19/2023

#include <vector>
#include <math.h>
#include "ML_node.h"
#include "ML_layer.h"
class ML_network
{
    public:
        ML_network(ML_nType, int);
    private:
         std::vector<ML_layer> layer_layer;
};