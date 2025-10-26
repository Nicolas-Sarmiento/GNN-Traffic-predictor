#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <Graph.hpp>

using json = nlohmann::json;

int main() {
    torch::Tensor tensor = torch::rand({2, 2});
    std::cout << "Random Tensor from LibTorch:\n" << tensor << std::endl;
    int i = 1;
    std::cout << i+1 << '\n';
    json data;
    data["name"] = "Traffic Prediction Model";
    data["version"] = 1.0;
    data["tensor_sample"] = { tensor[0][0].item<float>(), tensor[0][1].item<float>() };

    std::cout << "\nJSON Output:\n" << data.dump(4) << std::endl;

    std::ofstream file("output.json");
    file << data.dump(4);
    file.close();
    std::cout << "\nSaved JSON to output.json\n";

    return 0;
}
