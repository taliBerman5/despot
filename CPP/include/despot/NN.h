//
// Created by talib on 7/7/2022.
//

#ifndef POMCP_NN_H
#define POMCP_NN_H


#include <torch/torch.h>

class NN {
public:

    NN();

      struct Net : torch::nn::Module {
        Net(int num_states, int num_actions): nstates(num_states),  nactions(num_actions){
            // Construct and register two Linear submodules.
            fc1 = register_module("fc1", torch::nn::Linear(nstates, nstates));
            fc2 = register_module("fc2", torch::nn::Linear(nstates, nactions * 2));
        }

        // Implement the Net's algorithm.
        torch::Tensor forward(torch::Tensor x) {
            // Use one of many tensor manipulation functions.
            x = torch::relu(fc1->forward(x));
            x = fc2->forward(x);
//            x = torch::reshape(x, [-1, 2, nactions]);
//            x = x.view((-1, 2, nactions));  //TODO: check it is correct
//            x = torch::nn::UnflattenImpl(1, (2, nactions));
            return x;
        }

        // Use one of many "standard library" modules.
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
        int nstates;
        int nactions;
    };

    /**
 * output is what we predicted (v and n)
 * target is what we get
 */
    virtual int bsa_loss(torch::Tensor output, torch::Tensor target);
    virtual void train_model(Net model, int epoch);
    Net init_model(){
        Net model(3, 2);
        train_model(model, 3);
        return model;
    };

};


#endif //POMCP_NN_H
