// API created by William Shepelak and Carmen Thom 3/19/2023

/*
    This file containes the class used for each node of the ML network
    ...
*/

#include <vector>


class ML_node
{
    public:
        ML_node();
    private:
        double weights;
        double bias;
};