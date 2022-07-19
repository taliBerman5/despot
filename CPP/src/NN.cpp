//
// Created by talib on 7/7/2022.
//

#include "despot/NN.h"



NN::NN(){};
int //TODO
NN::bsa_loss(torch::Tensor output, torch::Tensor target) {
return 1;

}

void NN::train_model(Net model, int epoch) {
    torch::optim::Adam optimizer(model.parameters());
    std::cout << model.forward(torch::ones({3})) << std::endl;
}


