#include "lc0_weights.h"

#include "lc0_proto_wire.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace makaira::lc0 {

namespace {

constexpr std::uint32_t kWeightMagic = 0x1c0;

float u32_to_float(std::uint32_t x) {
    return std::bit_cast<float>(x);
}

float fp16_to_fp32(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x03FFu;

    std::uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            std::uint32_t m = mant;
            int e = -14;
            while ((m & 0x0400u) == 0u) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            bits = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        const std::uint32_t e = exp + (127 - 15);
        bits = sign | (e << 23) | (mant << 13);
    }
    return std::bit_cast<float>(bits);
}

float bf16_to_fp32(std::uint16_t b) {
    const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    return std::bit_cast<float>(bits);
}

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("lc0 weights: " + msg);
}

std::uint32_t get_u32_fixed(const FieldView& f) {
    return f.fixed32_value;
}

std::vector<std::uint32_t> parse_packed_varints(BytesView packed) {
    std::vector<std::uint32_t> out;
    std::size_t offset = 0;
    while (offset < packed.size) {
        std::uint64_t v = 0;
        if (!read_varint(packed, offset, v)) {
            fail("invalid packed varint stream");
        }
        out.push_back(static_cast<std::uint32_t>(v));
    }
    return out;
}

Layer parse_layer(BytesView msg) {
    Layer layer;
    std::vector<std::uint8_t> params;

    std::size_t offset = 0;
    FieldView f;
    while (next_field(msg, offset, f)) {
        if (f.number == 1 && f.wire_type == WireType::FIXED32) {
            layer.min_val = u32_to_float(get_u32_fixed(f));
        } else if (f.number == 2 && f.wire_type == WireType::FIXED32) {
            layer.max_val = u32_to_float(get_u32_fixed(f));
        } else if (f.number == 3 && f.wire_type == WireType::LENGTH_DELIMITED) {
            params.assign(f.bytes.data, f.bytes.data + f.bytes.size);
        } else if (f.number == 4 && f.wire_type == WireType::VARINT) {
            layer.encoding = static_cast<LayerEncoding>(static_cast<int>(f.varint_value));
        } else if (f.number == 5 && f.wire_type == WireType::VARINT) {
            layer.dims.push_back(static_cast<std::uint32_t>(f.varint_value));
        } else if (f.number == 5 && f.wire_type == WireType::LENGTH_DELIMITED) {
            auto dims = parse_packed_varints(f.bytes);
            layer.dims.insert(layer.dims.end(), dims.begin(), dims.end());
        }
    }

    if (layer.encoding == LayerEncoding::UNKNOWN) {
        layer.encoding = LayerEncoding::LINEAR16;
    }

    if (params.empty()) {
        return layer;
    }

    switch (layer.encoding) {
        case LayerEncoding::LINEAR16: {
            if ((params.size() & 1u) != 0u) {
                fail("LINEAR16 layer has odd byte size");
            }
            const float lo = layer.min_val;
            const float hi = layer.max_val;
            const std::size_t n = params.size() / 2;
            layer.values.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint16_t u = static_cast<std::uint16_t>(params[2 * i])
                                      | (static_cast<std::uint16_t>(params[2 * i + 1]) << 8);
                const float theta = static_cast<float>(u) / 65535.0f;
                layer.values[i] = lo * (1.0f - theta) + hi * theta;
            }
            break;
        }
        case LayerEncoding::FLOAT16: {
            if ((params.size() & 1u) != 0u) {
                fail("FLOAT16 layer has odd byte size");
            }
            const std::size_t n = params.size() / 2;
            layer.values.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint16_t u = static_cast<std::uint16_t>(params[2 * i])
                                      | (static_cast<std::uint16_t>(params[2 * i + 1]) << 8);
                layer.values[i] = fp16_to_fp32(u);
            }
            break;
        }
        case LayerEncoding::BFLOAT16: {
            if ((params.size() & 1u) != 0u) {
                fail("BFLOAT16 layer has odd byte size");
            }
            const std::size_t n = params.size() / 2;
            layer.values.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint16_t u = static_cast<std::uint16_t>(params[2 * i])
                                      | (static_cast<std::uint16_t>(params[2 * i + 1]) << 8);
                layer.values[i] = bf16_to_fp32(u);
            }
            break;
        }
        case LayerEncoding::FLOAT32: {
            if ((params.size() & 3u) != 0u) {
                fail("FLOAT32 layer byte size is not multiple of 4");
            }
            const std::size_t n = params.size() / 4;
            layer.values.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                std::uint32_t u = static_cast<std::uint32_t>(params[4 * i])
                                | (static_cast<std::uint32_t>(params[4 * i + 1]) << 8)
                                | (static_cast<std::uint32_t>(params[4 * i + 2]) << 16)
                                | (static_cast<std::uint32_t>(params[4 * i + 3]) << 24);
                layer.values[i] = std::bit_cast<float>(u);
            }
            break;
        }
        default:
            fail("unsupported layer encoding " + std::to_string(static_cast<int>(layer.encoding)));
    }

    return layer;
}

Smolgen parse_smolgen(BytesView msg) {
    Smolgen s;
    s.present = true;
    if (auto x = first_submessage(msg, 1)) s.compress = parse_layer(*x);
    if (auto x = first_submessage(msg, 2)) s.dense1_w = parse_layer(*x);
    if (auto x = first_submessage(msg, 3)) s.dense1_b = parse_layer(*x);
    if (auto x = first_submessage(msg, 4)) s.ln1_g = parse_layer(*x);
    if (auto x = first_submessage(msg, 5)) s.ln1_b = parse_layer(*x);
    if (auto x = first_submessage(msg, 6)) s.dense2_w = parse_layer(*x);
    if (auto x = first_submessage(msg, 7)) s.dense2_b = parse_layer(*x);
    if (auto x = first_submessage(msg, 8)) s.ln2_g = parse_layer(*x);
    if (auto x = first_submessage(msg, 9)) s.ln2_b = parse_layer(*x);
    return s;
}

Mha parse_mha(BytesView msg) {
    Mha m;
    if (auto x = first_submessage(msg, 1)) m.q_w = parse_layer(*x);
    if (auto x = first_submessage(msg, 2)) m.q_b = parse_layer(*x);
    if (auto x = first_submessage(msg, 3)) m.k_w = parse_layer(*x);
    if (auto x = first_submessage(msg, 4)) m.k_b = parse_layer(*x);
    if (auto x = first_submessage(msg, 5)) m.v_w = parse_layer(*x);
    if (auto x = first_submessage(msg, 6)) m.v_b = parse_layer(*x);
    if (auto x = first_submessage(msg, 7)) m.dense_w = parse_layer(*x);
    if (auto x = first_submessage(msg, 8)) m.dense_b = parse_layer(*x);
    if (auto x = first_submessage(msg, 9)) m.smolgen = parse_smolgen(*x);
    return m;
}

Ffn parse_ffn(BytesView msg) {
    Ffn f;
    if (auto x = first_submessage(msg, 1)) f.dense1_w = parse_layer(*x);
    if (auto x = first_submessage(msg, 2)) f.dense1_b = parse_layer(*x);
    if (auto x = first_submessage(msg, 3)) f.dense2_w = parse_layer(*x);
    if (auto x = first_submessage(msg, 4)) f.dense2_b = parse_layer(*x);
    return f;
}

EncoderLayer parse_encoder(BytesView msg) {
    EncoderLayer e;
    if (auto x = first_submessage(msg, 1)) e.mha = parse_mha(*x);
    if (auto x = first_submessage(msg, 2)) e.ln1_g = parse_layer(*x);
    if (auto x = first_submessage(msg, 3)) e.ln1_b = parse_layer(*x);
    if (auto x = first_submessage(msg, 4)) e.ffn = parse_ffn(*x);
    if (auto x = first_submessage(msg, 5)) e.ln2_g = parse_layer(*x);
    if (auto x = first_submessage(msg, 6)) e.ln2_b = parse_layer(*x);
    return e;
}

void normalize_network_format(Weights& w) {
    auto& nf = w.format;

    if (!nf.has_network_format) {
        nf.input_format = 1;
        nf.output_format = 1;
        nf.network_structure = 3;
        nf.value_format = 1;
        nf.policy_format = 1;
    } else if (nf.network_structure == 1) {
        nf.network_structure = 3;
        nf.value_format = 1;
        nf.policy_format = 1;
    } else if (nf.network_structure == 2) {
        nf.network_structure = 4;
        nf.value_format = 1;
        nf.policy_format = 1;
    } else if (nf.network_structure == 4 && !w.encoders.empty()) {
        nf.network_structure = 6;
        if (w.has_smolgen_global) {
            nf.ffn_activation = static_cast<int>(Activation::RELU2);
            nf.smolgen_activation = static_cast<int>(Activation::SWISH);
            nf.has_ffn_activation = true;
            nf.has_smolgen_activation = true;
        }
    } else if (nf.network_structure == 134) {
        nf.network_structure = 7;
    }

    if (nf.network_structure == 6 && !nf.has_input_embedding) {
        nf.input_embedding = 1;  // INPUT_EMBEDDING_PE_MAP
        nf.has_input_embedding = true;
    }
}

}  // namespace

Weights load_from_pb_gz(const std::string& path) {
    const auto bytes = read_gzip_file(path);
    const BytesView net{bytes.data(), bytes.size()};

    Weights out;

    if (auto magic = first_field(net, 1, WireType::FIXED32)) {
        out.magic = magic->fixed32_value;
    }
    if (out.magic != kWeightMagic) {
        fail("bad magic header in " + path);
    }

    if (auto fmt = first_submessage(net, 4)) {
        if (auto nf = first_submessage(*fmt, 2)) {
            out.format.has_network_format = true;
            if (auto f = first_field(*nf, 1, WireType::VARINT)) out.format.input_format = static_cast<int>(f->varint_value);
            if (auto f = first_field(*nf, 2, WireType::VARINT)) out.format.output_format = static_cast<int>(f->varint_value);
            if (auto f = first_field(*nf, 3, WireType::VARINT)) out.format.network_structure = static_cast<int>(f->varint_value);
            if (auto f = first_field(*nf, 4, WireType::VARINT)) out.format.policy_format = static_cast<int>(f->varint_value);
            if (auto f = first_field(*nf, 5, WireType::VARINT)) out.format.value_format = static_cast<int>(f->varint_value);
            if (auto f = first_field(*nf, 6, WireType::VARINT)) out.format.moves_left_format = static_cast<int>(f->varint_value);
            if (auto f = first_field(*nf, 7, WireType::VARINT)) out.format.default_activation = static_cast<int>(f->varint_value);
            if (auto f = first_field(*nf, 8, WireType::VARINT)) {
                out.format.smolgen_activation = static_cast<int>(f->varint_value);
                out.format.has_smolgen_activation = true;
            }
            if (auto f = first_field(*nf, 9, WireType::VARINT)) {
                out.format.ffn_activation = static_cast<int>(f->varint_value);
                out.format.has_ffn_activation = true;
            }
            if (auto f = first_field(*nf, 10, WireType::VARINT)) {
                out.format.input_embedding = static_cast<int>(f->varint_value);
                out.format.has_input_embedding = true;
            }
        }
    }

    const auto weights_msg = first_submessage(net, 10);
    if (!weights_msg.has_value()) {
        fail("missing weights message");
    }

    const BytesView w = *weights_msg;

    if (auto f = first_field(w, 28, WireType::VARINT)) out.headcount = static_cast<int>(f->varint_value);

    if (auto x = first_submessage(w, 25)) out.ip_emb_w = parse_layer(*x);
    if (auto x = first_submessage(w, 26)) out.ip_emb_b = parse_layer(*x);
    if (auto x = first_submessage(w, 33)) out.ip_mult_gate = parse_layer(*x);
    if (auto x = first_submessage(w, 34)) out.ip_add_gate = parse_layer(*x);
    if (auto x = first_submessage(w, 35)) {
        out.smolgen_w = parse_layer(*x);
        out.has_smolgen_global = !out.smolgen_w.values.empty();
    }

    for (const auto& enc_msg : all_submessages(w, 27)) {
        out.encoders.push_back(parse_encoder(enc_msg));
    }

    if (auto x = first_submessage(w, 29)) out.ip_val_w = parse_layer(*x);
    if (auto x = first_submessage(w, 30)) out.ip_val_b = parse_layer(*x);
    if (auto x = first_submessage(w, 7)) out.ip1_val_w = parse_layer(*x);
    if (auto x = first_submessage(w, 8)) out.ip1_val_b = parse_layer(*x);
    if (auto x = first_submessage(w, 9)) out.ip2_val_w = parse_layer(*x);
    if (auto x = first_submessage(w, 10)) out.ip2_val_b = parse_layer(*x);

    if (auto x = first_submessage(w, 4)) out.ip_pol_w = parse_layer(*x);
    if (auto x = first_submessage(w, 5)) out.ip_pol_b = parse_layer(*x);
    if (auto x = first_submessage(w, 17)) out.ip2_pol_w = parse_layer(*x);
    if (auto x = first_submessage(w, 18)) out.ip2_pol_b = parse_layer(*x);
    if (auto x = first_submessage(w, 19)) out.ip3_pol_w = parse_layer(*x);
    if (auto x = first_submessage(w, 20)) out.ip3_pol_b = parse_layer(*x);
    if (auto x = first_submessage(w, 22)) out.ip4_pol_w = parse_layer(*x);

    normalize_network_format(out);
    return out;
}

int layer_output_size(const Layer& w, const Layer& b, const std::string& name) {
    const int out = static_cast<int>(b.values.size());
    if (out <= 0) {
        fail(name + ": bias vector is empty");
    }
    if (w.values.empty()) {
        fail(name + ": weight vector is empty");
    }
    if (static_cast<int>(w.values.size()) % out != 0) {
        fail(name + ": weight size " + std::to_string(w.values.size())
             + " not divisible by output size " + std::to_string(out));
    }
    return out;
}

int layer_input_size(const Layer& w, const Layer& b, const std::string& name) {
    const int out = layer_output_size(w, b, name);
    return static_cast<int>(w.values.size()) / out;
}

void validate_attention_value_shapes(const Weights& w, bool strict_t1_shape) {
    if (w.format.input_format != 1) {
        fail("input format must be INPUT_CLASSICAL_112_PLANE (1)");
    }
    if (w.format.value_format != 2) {
        fail("value format must be VALUE_WDL (2)");
    }
    if (w.format.network_structure != 6 && w.format.network_structure != 7) {
        fail("network structure must be attention-body format after normalization");
    }

    const int embedding = static_cast<int>(w.ip_emb_b.values.size());
    if (embedding <= 0) {
        fail("ip_emb_b is empty");
    }
    if (w.headcount <= 0) {
        fail("headcount must be > 0");
    }
    if (embedding % w.headcount != 0) {
        fail("embedding size " + std::to_string(embedding)
             + " is not divisible by headcount " + std::to_string(w.headcount));
    }

    const int ip_emb_in = layer_input_size(w.ip_emb_w, w.ip_emb_b, "ip_emb");
    if (ip_emb_in != 176) {
        fail("ip_emb input size expected 176, got " + std::to_string(ip_emb_in));
    }

    if (w.encoders.empty()) {
        fail("encoder list is empty");
    }

    if (strict_t1_shape) {
        if (static_cast<int>(w.encoders.size()) != 10) {
            fail("expected encoder_layers == 10 for t1 net, got " + std::to_string(w.encoders.size()));
        }
        if (embedding != 256) {
            fail("expected embedding == 256 for t1 net, got " + std::to_string(embedding));
        }
        if (w.headcount != 8) {
            fail("expected headcount == 8 for t1 net, got " + std::to_string(w.headcount));
        }
    }

    for (std::size_t i = 0; i < w.encoders.size(); ++i) {
        const auto& e = w.encoders[i];
        const std::string p = "encoder[" + std::to_string(i) + "]";

        const int q_out = layer_output_size(e.mha.q_w, e.mha.q_b, p + ".q");
        const int q_in = layer_input_size(e.mha.q_w, e.mha.q_b, p + ".q");
        const int k_out = layer_output_size(e.mha.k_w, e.mha.k_b, p + ".k");
        const int v_out = layer_output_size(e.mha.v_w, e.mha.v_b, p + ".v");
        const int d_out = layer_output_size(e.mha.dense_w, e.mha.dense_b, p + ".dense");
        const int d_in = layer_input_size(e.mha.dense_w, e.mha.dense_b, p + ".dense");

        if (q_in != embedding || q_out != embedding || k_out != embedding || v_out != embedding) {
            fail(p + ": MHA projection dimensions must all be embedding-sized");
        }
        if (d_in != embedding || d_out != embedding) {
            fail(p + ": MHA output projection must be embedding->embedding");
        }

        const int f1_out = layer_output_size(e.ffn.dense1_w, e.ffn.dense1_b, p + ".ffn1");
        const int f1_in = layer_input_size(e.ffn.dense1_w, e.ffn.dense1_b, p + ".ffn1");
        const int f2_out = layer_output_size(e.ffn.dense2_w, e.ffn.dense2_b, p + ".ffn2");
        const int f2_in = layer_input_size(e.ffn.dense2_w, e.ffn.dense2_b, p + ".ffn2");

        if (f1_in != embedding || f2_out != embedding || f2_in != f1_out) {
            fail(p + ": FFN dimensions must be embedding->dff->embedding");
        }

        if (e.mha.smolgen.present) {
            const int c_out = layer_output_size(e.mha.smolgen.compress, e.mha.q_b, p + ".smolgen.compress");
            (void)c_out;
            if (w.smolgen_w.values.empty()) {
                fail(p + ": smolgen present in layer but global smolgen_w missing");
            }
        }
    }

    const int val_tok_out = layer_output_size(w.ip_val_w, w.ip_val_b, "ip_val");
    const int val_tok_in = layer_input_size(w.ip_val_w, w.ip_val_b, "ip_val");
    if (val_tok_in != embedding) {
        fail("ip_val input must equal embedding");
    }

    const int val1_out = layer_output_size(w.ip1_val_w, w.ip1_val_b, "ip1_val");
    const int val1_in = layer_input_size(w.ip1_val_w, w.ip1_val_b, "ip1_val");
    if (val1_in != val_tok_out * 64) {
        fail("ip1_val input must equal 64 * ip_val_out");
    }

    const int val2_out = layer_output_size(w.ip2_val_w, w.ip2_val_b, "ip2_val");
    const int val2_in = layer_input_size(w.ip2_val_w, w.ip2_val_b, "ip2_val");
    if (val2_in != val1_out) {
        fail("ip2_val input must equal ip1_val output");
    }
    if (val2_out != 3) {
        fail("WDL head output size must be exactly 3");
    }
}

}  // namespace makaira::lc0
