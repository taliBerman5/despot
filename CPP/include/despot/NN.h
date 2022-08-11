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
            x = torch::nn::Unflatten(1,std::vector<int64_t>{2, nactions})(x);
            std::cout << "x"<< std::endl;
            std::cout << x << std::endl;
            std::cout << "x.select(1, 1)"<< std::endl;
            std::cout << x.select(1, 1)<< std::endl;
            std::cout << "************size*************" << std::endl;
            std::cout << x.sizes() << std::endl;
            return x;
        }

        // Use one of many "standard library" modules.
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
        int nstates;
        int nactions;
    };


    virtual int bsa_loss(torch::Tensor output, torch::Tensor target);
    virtual void train_model(Net model, int epoch);
    Net init_model(){
        Net model(3, 5);
        train_model(model, 3);
        return model;
    };

//    std::vector<int> get_tensor_shape(torch::Tensor& tensor)
//    {
//        std::vector<int> shape;
//        int num_dimensions = get_tensor_shape(tensor).dims();
//        for(int ii_dim=0; ii_dim<num_dimensions; ii_dim++) {
//            shape.push_back(tensor.shape().dim_size(ii_dim));
//        }
//        return shape;
//    }


};


#endif //POMCP_NN_H
