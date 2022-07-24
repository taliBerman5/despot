//
// Created by talib on 7/7/2022.
//

#include "despot/NN.h"



NN::NN(){};

/**
* output is what we predicted (v and n)
* target is what we get
*/
int NN::bsa_loss(torch::Tensor output, torch::Tensor target) {  //TODO

return 1;

}

void NN::train_model(Net model, int epoch) {
    torch::optim::Adam optimizer(model.parameters());
    std::cout << model.forward(torch::ones({10, 3})) << std::endl;
}


