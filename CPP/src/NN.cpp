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
    torch::Tensor logitmean = output.select(1, 0);
    torch::Tensor logn = output.select(1, 1);
    torch::Tensor mean = torch::special::expit(logitmean);
    torch::Tensor n = torch::exp(logn);
    torch::Tensor v = target.select(1, 0);
    torch::Tensor ss = target.select(1, 1);
    torch::Tensor neff = 1/(1/n + 1/ss);
    torch::Tensor alpha = mean * neff;
    torch::Tensor beta = neff - alpha;
    return torch::mean(torch::beta::Beta(alpha, beta));

}

torch::Tensor log_prob(torch::Tensor x, const torch::Tensor &value) {
    //TODO:check if correct. The var, mean, log_std suppose to be on x? the calculation is correct of lz and the return?
    double lz = log(sqrt(2 * M_PI));
    torch::Tensor var = torch::pow(2, x.std());
    torch::Tensor log_std = x.std().log();
    torch::Tensor mean = torch::mean(x);
    return -(value - mean)*(value - mean) / (2 * var) - log_std - lz;
}

void NN::train_model(Net model, int epoch) {
//    torch::Tensor t = torch::ones({10, 3});
    torch::Tensor t = torch::tensor({{1,2,3},{4,5,6}, {7,8,9}, {10,11,12}});
    t = t.select(1, 0);
    std::cout << t<< std::endl;
    std::cout << "t.slice(1) "<< std::endl;
    std::cout << t.sizes() << std::endl;
    torch::optim::Adam optimizer(model.parameters());
    std::cout << model.forward(torch::ones({4, 3})) << std::endl;
}


