// Check frozen LR/XGB/StateMLP/ET exports without loading q-v vectors.
#include "online_gate.h"
#include <fstream>
#include <iostream>
#include <cmath>

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    const std::string folder = argv[1];
    const unsigned dim = unsigned(std::stoul(argv[2]));
    diskann::OnlineGate lr(folder+"/lr.bin", "", 0),
        xgb(folder+"/xgb.bin", "", 0), state(folder+"/state.bin", "", 0),
        et(folder+"/et.bin", "", 0);
    diskann::OnlineGate *models[] = {&lr, &xgb, &state};
    std::ifstream check(folder+"/native_check.bin", std::ios::binary);
    uint32_t n = 0;
    check.read(reinterpret_cast<char*>(&n), 4);
    if (!check || !n || n > 1000000) return 3;
    std::vector<float> row(dim+22);
    float error[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < n; ++i) {
        check.read(reinterpret_cast<char*>(row.data()), row.size()*4);
        if (!check) return 3;
        for (unsigned j = 0; j < 3; ++j) {
            // The stored node field is unused by these state-only models.
            float score = models[j]->score(row.data(), 0, row.data()+dim+1);
            float expected = row[dim+18+j];
            if (!std::isfinite(score) || !std::isfinite(expected)) return 4;
            error[j] = std::max(error[j], std::abs(score-expected));
        }
    }
    std::ifstream etcheck(folder+"/et_check.bin", std::ios::binary);
    uint32_t ne = 0;
    etcheck.read(reinterpret_cast<char*>(&ne), 4);
    if (!etcheck || !ne || ne > 1000000) return 3;
    float erow[18], query[256] = {};
    for (uint32_t i = 0; i < ne; ++i) {
        etcheck.read(reinterpret_cast<char*>(erow), sizeof(erow));
        if (!etcheck) return 3;
        float score = et.score(query, 0, erow);
        if (!std::isfinite(score) || !std::isfinite(erow[17])) return 4;
        error[3] = std::max(error[3], std::abs(score-erow[17]));
    }
    std::cout << "{\"LR\":" << error[0] << ",\"XGB\":" << error[1]
              << ",\"MLP\":" << error[2] << ",\"ET\":" << error[3]
              << ",\"samples\":" << n << ",\"et_samples\":" << ne << "}\n";
    for (float e : error) if (e > 2e-5f) return 4;
    return 0;
}
