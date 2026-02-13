#pragma once

#include "lc0_proto_wire.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace makaira::lc0 {

enum class LayerEncoding : int {
    UNKNOWN = 0,
    LINEAR16 = 1,
    FLOAT16 = 2,
    BFLOAT16 = 3,
    FLOAT32 = 4,
};

enum class Activation : int {
    DEFAULT = 0,
    MISH = 1,
    RELU = 2,
    NONE = 3,
    TANH = 4,
    SIGMOID = 5,
    SELU = 6,
    SWISH = 7,
    RELU2 = 8,
    SOFTMAX = 9,
};

struct Layer {
    float min_val = 0.0f;
    float max_val = 0.0f;
    LayerEncoding encoding = LayerEncoding::LINEAR16;
    std::vector<std::uint32_t> dims{};
    std::vector<float> values{};
};

struct Smolgen {
    Layer compress;
    Layer dense1_w;
    Layer dense1_b;
    Layer ln1_g;
    Layer ln1_b;
    Layer dense2_w;
    Layer dense2_b;
    Layer ln2_g;
    Layer ln2_b;
    bool present = false;
};

struct Mha {
    Layer q_w;
    Layer q_b;
    Layer k_w;
    Layer k_b;
    Layer v_w;
    Layer v_b;
    Layer dense_w;
    Layer dense_b;
    Smolgen smolgen;
};

struct Ffn {
    Layer dense1_w;
    Layer dense1_b;
    Layer dense2_w;
    Layer dense2_b;
};

struct EncoderLayer {
    Mha mha;
    Layer ln1_g;
    Layer ln1_b;
    Ffn ffn;
    Layer ln2_g;
    Layer ln2_b;
};

struct NetworkFormat {
    int input_format = 0;
    int output_format = 0;
    int network_structure = 0;
    int policy_format = 0;
    int value_format = 0;
    int moves_left_format = 0;
    int default_activation = 0;
    int ffn_activation = 0;
    int smolgen_activation = 0;
    int input_embedding = 0;
    bool has_network_format = false;
    bool has_ffn_activation = false;
    bool has_smolgen_activation = false;
    bool has_input_embedding = false;
};

struct Weights {
    std::uint32_t magic = 0;
    NetworkFormat format{};

    Layer ip_emb_w;
    Layer ip_emb_b;
    Layer ip_mult_gate;
    Layer ip_add_gate;
    Layer smolgen_w;

    std::vector<EncoderLayer> encoders{};
    int headcount = 0;

    Layer ip_val_w;
    Layer ip_val_b;
    Layer ip1_val_w;
    Layer ip1_val_b;
    Layer ip2_val_w;
    Layer ip2_val_b;

    Layer ip_pol_w;
    Layer ip_pol_b;
    Layer ip2_pol_w;
    Layer ip2_pol_b;
    Layer ip3_pol_w;
    Layer ip3_pol_b;
    Layer ip4_pol_w;

    bool has_smolgen_global = false;
};

Weights load_from_pb_gz(const std::string& path);

void validate_attention_value_shapes(const Weights& w, bool strict_t1_shape = true);

int layer_output_size(const Layer& w, const Layer& b, const std::string& name);
int layer_input_size(const Layer& w, const Layer& b, const std::string& name);

}  // namespace makaira::lc0
