#include "lc0_attention_value.h"

#include "lc0_activation.h"
#include "lc0_tables.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace makaira::lc0 {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("lc0 forward: " + msg);
}

Activation default_activation(const Weights& w) {
    return w.format.default_activation == 1 ? Activation::MISH : Activation::RELU;
}

Activation resolve_activation(const Weights& w, int encoded, bool has_specific) {
    if (!has_specific || encoded == static_cast<int>(Activation::DEFAULT)) {
        return default_activation(w);
    }
    return static_cast<Activation>(encoded);
}

int infer_out_dim_no_bias(const Layer& w, int in_dim, const std::string& name) {
    if (in_dim <= 0) {
        fail(name + ": in_dim must be > 0");
    }
    if (w.values.empty()) {
        fail(name + ": weights are empty");
    }
    if (static_cast<int>(w.values.size()) % in_dim != 0) {
        fail(name + ": weight size not divisible by in_dim");
    }
    return static_cast<int>(w.values.size()) / in_dim;
}

void fc_rows(const std::vector<float>& in,
             int rows,
             int in_dim,
             const Layer& w,
             const Layer& b,
             Activation act,
             const std::string& name,
             std::vector<float>& out) {
    const int out_dim = layer_output_size(w, b, name);
    const int inferred_in = layer_input_size(w, b, name);
    if (inferred_in != in_dim) {
        fail(name + ": expected in_dim " + std::to_string(inferred_in) + ", got " + std::to_string(in_dim));
    }
    if (static_cast<int>(in.size()) != rows * in_dim) {
        fail(name + ": input vector size mismatch");
    }

    out.assign(static_cast<std::size_t>(rows * out_dim), 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float* x = in.data() + static_cast<std::size_t>(r * in_dim);
        float* y = out.data() + static_cast<std::size_t>(r * out_dim);

        for (int o = 0; o < out_dim; ++o) {
            const float* ww = w.values.data() + static_cast<std::size_t>(o * in_dim);
            float sum = b.values[static_cast<std::size_t>(o)];
            for (int i = 0; i < in_dim; ++i) {
                sum += ww[i] * x[i];
            }
            y[o] = activate_scalar(sum, act);
        }
    }
}

void fc_rows_no_bias(const std::vector<float>& in,
                     int rows,
                     int in_dim,
                     const Layer& w,
                     Activation act,
                     const std::string& name,
                     std::vector<float>& out) {
    const int out_dim = infer_out_dim_no_bias(w, in_dim, name);
    if (static_cast<int>(in.size()) != rows * in_dim) {
        fail(name + ": input vector size mismatch");
    }

    out.assign(static_cast<std::size_t>(rows * out_dim), 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float* x = in.data() + static_cast<std::size_t>(r * in_dim);
        float* y = out.data() + static_cast<std::size_t>(r * out_dim);

        for (int o = 0; o < out_dim; ++o) {
            const float* ww = w.values.data() + static_cast<std::size_t>(o * in_dim);
            float sum = 0.0f;
            for (int i = 0; i < in_dim; ++i) {
                sum += ww[i] * x[i];
            }
            y[o] = activate_scalar(sum, act);
        }
    }
}

void layer_norm_skip(std::vector<float>& data,
                     const std::vector<float>* skip,
                     int rows,
                     int channels,
                     float alpha,
                     const Layer& gammas,
                     const Layer& betas,
                     float eps,
                     const std::string& name) {
    if (static_cast<int>(gammas.values.size()) != channels || static_cast<int>(betas.values.size()) != channels) {
        fail(name + ": ln gamma/beta size mismatch");
    }

    for (int r = 0; r < rows; ++r) {
        float mean = 0.0f;
        for (int c = 0; c < channels; ++c) {
            const std::size_t idx = static_cast<std::size_t>(r * channels + c);
            float v = data[idx] * alpha;
            if (skip) {
                v += (*skip)[idx];
            }
            data[idx] = v;
            mean += v;
        }
        mean /= static_cast<float>(channels);

        float var = 0.0f;
        for (int c = 0; c < channels; ++c) {
            const std::size_t idx = static_cast<std::size_t>(r * channels + c);
            const float d = data[idx] - mean;
            var += d * d;
        }
        var /= static_cast<float>(channels);

        const float inv = 1.0f / std::sqrt(var + eps);
        for (int c = 0; c < channels; ++c) {
            const std::size_t idx = static_cast<std::size_t>(r * channels + c);
            data[idx] = betas.values[static_cast<std::size_t>(c)]
                        + gammas.values[static_cast<std::size_t>(c)] * (data[idx] - mean) * inv;
        }
    }
}

void add_smolgen_bias(const Weights& w,
                      const EncoderLayer& layer,
                      const std::vector<float>& x,
                      int embedding,
                      int heads,
                      std::vector<float>& scores) {
    if (!layer.mha.smolgen.present) {
        return;
    }

    const Smolgen& sg = layer.mha.smolgen;

    std::vector<float> compressed;
    fc_rows_no_bias(x, 64, embedding, sg.compress, Activation::NONE, "smolgen.compress", compressed);
    const int hidden_channels = static_cast<int>(compressed.size() / 64);

    std::vector<float> compressed_flat = compressed;

    std::vector<float> dense1;
    fc_rows(compressed_flat,
            1,
            64 * hidden_channels,
            sg.dense1_w,
            sg.dense1_b,
            resolve_activation(w, w.format.smolgen_activation, w.format.has_smolgen_activation),
            "smolgen.dense1",
            dense1);
    layer_norm_skip(dense1, nullptr, 1, static_cast<int>(dense1.size()), 1.0f, sg.ln1_g, sg.ln1_b, 1e-3f, "smolgen.ln1");

    std::vector<float> dense2;
    fc_rows(dense1,
            1,
            static_cast<int>(dense1.size()),
            sg.dense2_w,
            sg.dense2_b,
            resolve_activation(w, w.format.smolgen_activation, w.format.has_smolgen_activation),
            "smolgen.dense2",
            dense2);
    layer_norm_skip(dense2, nullptr, 1, static_cast<int>(dense2.size()), 1.0f, sg.ln2_g, sg.ln2_b, 1e-3f, "smolgen.ln2");

    if (w.smolgen_w.values.empty()) {
        fail("global smolgen_w is empty while encoder smolgen is present");
    }

    const int per_head = static_cast<int>(dense2.size()) / heads;
    if (per_head <= 0 || per_head * heads != static_cast<int>(dense2.size())) {
        fail("smolgen dense2 size is not divisible by headcount");
    }

    const int smolgen_out = infer_out_dim_no_bias(w.smolgen_w, per_head, "global smolgen_w");
    if (smolgen_out != 64 * 64) {
        fail("global smolgen_w output must be 4096");
    }

    for (int h = 0; h < heads; ++h) {
        std::vector<float> in_head(static_cast<std::size_t>(per_head));
        for (int i = 0; i < per_head; ++i) {
            in_head[static_cast<std::size_t>(i)] = dense2[static_cast<std::size_t>(h * per_head + i)];
        }

        std::vector<float> out_head;
        fc_rows_no_bias(in_head, 1, per_head, w.smolgen_w, Activation::NONE, "global smolgen apply", out_head);

        for (int q = 0; q < 64; ++q) {
            for (int k = 0; k < 64; ++k) {
                scores[static_cast<std::size_t>(h * 64 * 64 + q * 64 + k)] += out_head[static_cast<std::size_t>(q * 64 + k)];
            }
        }
    }
}

}  // namespace

WdlOutput forward_attention_value(const Weights& w, const InputPlanes112& input) {
    validate_attention_value_shapes(w, false);

    const int embedding = static_cast<int>(w.ip_emb_b.values.size());
    const int heads = w.headcount;
    const int depth = embedding / heads;

    std::vector<float> token_in(static_cast<std::size_t>(64 * 176), 0.0f);
    for (int sq = 0; sq < 64; ++sq) {
        float* row = token_in.data() + static_cast<std::size_t>(sq * 176);
        for (int p = 0; p < 112; ++p) {
            row[p] = input[static_cast<std::size_t>(p * 64 + sq)];
        }
        for (int pe = 0; pe < 64; ++pe) {
            row[112 + pe] = k_pos_encoding[static_cast<std::size_t>(sq)][static_cast<std::size_t>(pe)];
        }
    }

    std::vector<float> x;
    fc_rows(token_in, 64, 176, w.ip_emb_w, w.ip_emb_b, default_activation(w), "ip_emb", x);

    if (!w.ip_mult_gate.values.empty() && !w.ip_add_gate.values.empty()) {
        if (static_cast<int>(w.ip_mult_gate.values.size()) != embedding * 64
            || static_cast<int>(w.ip_add_gate.values.size()) != embedding * 64) {
            fail("input gating vectors must have embedding*64 values");
        }
        for (int sq = 0; sq < 64; ++sq) {
            for (int c = 0; c < embedding; ++c) {
                const std::size_t xidx = static_cast<std::size_t>(sq * embedding + c);
                const std::size_t gidx = static_cast<std::size_t>(c * 64 + sq);
                x[xidx] = x[xidx] * w.ip_mult_gate.values[gidx] + w.ip_add_gate.values[gidx];
            }
        }
    }

    const float alpha = std::pow(2.0f * static_cast<float>(w.encoders.size()), -0.25f);
    const Activation ffn_act = resolve_activation(w, w.format.ffn_activation, w.format.has_ffn_activation);

    for (std::size_t li = 0; li < w.encoders.size(); ++li) {
        const EncoderLayer& layer = w.encoders[li];

        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        fc_rows(x, 64, embedding, layer.mha.q_w, layer.mha.q_b, Activation::NONE, "encoder.q", q);
        fc_rows(x, 64, embedding, layer.mha.k_w, layer.mha.k_b, Activation::NONE, "encoder.k", k);
        fc_rows(x, 64, embedding, layer.mha.v_w, layer.mha.v_b, Activation::NONE, "encoder.v", v);

        std::vector<float> scores(static_cast<std::size_t>(heads * 64 * 64), 0.0f);
        add_smolgen_bias(w, layer, x, embedding, heads, scores);

        const float scale = 1.0f / std::sqrt(static_cast<float>(depth));

        for (int h = 0; h < heads; ++h) {
            for (int qi = 0; qi < 64; ++qi) {
                for (int ki = 0; ki < 64; ++ki) {
                    float dot = 0.0f;
                    for (int d = 0; d < depth; ++d) {
                        const int c = h * depth + d;
                        dot += q[static_cast<std::size_t>(qi * embedding + c)]
                             * k[static_cast<std::size_t>(ki * embedding + c)];
                    }
                    scores[static_cast<std::size_t>(h * 64 * 64 + qi * 64 + ki)] += dot * scale;
                }
                softmax_inplace(scores.data() + static_cast<std::size_t>(h * 64 * 64 + qi * 64), 64);
            }
        }

        std::vector<float> attn(static_cast<std::size_t>(64 * embedding), 0.0f);
        for (int h = 0; h < heads; ++h) {
            for (int qi = 0; qi < 64; ++qi) {
                for (int ki = 0; ki < 64; ++ki) {
                    const float a = scores[static_cast<std::size_t>(h * 64 * 64 + qi * 64 + ki)];
                    for (int d = 0; d < depth; ++d) {
                        const int c = h * depth + d;
                        attn[static_cast<std::size_t>(qi * embedding + c)] +=
                          a * v[static_cast<std::size_t>(ki * embedding + c)];
                    }
                }
            }
        }

        std::vector<float> proj;
        fc_rows(attn, 64, embedding, layer.mha.dense_w, layer.mha.dense_b, Activation::NONE, "encoder.proj", proj);
        layer_norm_skip(proj, &x, 64, embedding, alpha, layer.ln1_g, layer.ln1_b, 1e-6f, "encoder.ln1");
        x.swap(proj);

        std::vector<float> ffn1;
        fc_rows(x,
                64,
                embedding,
                layer.ffn.dense1_w,
                layer.ffn.dense1_b,
                ffn_act,
                "encoder.ffn1",
                ffn1);
        const int dff = static_cast<int>(ffn1.size() / 64);

        std::vector<float> ffn2;
        fc_rows(ffn1,
                64,
                dff,
                layer.ffn.dense2_w,
                layer.ffn.dense2_b,
                Activation::NONE,
                "encoder.ffn2",
                ffn2);
        layer_norm_skip(ffn2, &x, 64, embedding, alpha, layer.ln2_g, layer.ln2_b, 1e-6f, "encoder.ln2");
        x.swap(ffn2);
    }

    std::vector<float> val_tokens;
    fc_rows(x, 64, embedding, w.ip_val_w, w.ip_val_b, default_activation(w), "ip_val", val_tokens);

    const int val_planes = static_cast<int>(val_tokens.size() / 64);
    std::vector<float> val_flat(static_cast<std::size_t>(64 * val_planes), 0.0f);
    for (int i = 0; i < 64 * val_planes; ++i) {
        val_flat[static_cast<std::size_t>(i)] = val_tokens[static_cast<std::size_t>(i)];
    }

    std::vector<float> val1;
    fc_rows(val_flat,
            1,
            64 * val_planes,
            w.ip1_val_w,
            w.ip1_val_b,
            default_activation(w),
            "ip1_val",
            val1);

    std::vector<float> val2;
    fc_rows(val1,
            1,
            static_cast<int>(val1.size()),
            w.ip2_val_w,
            w.ip2_val_b,
            Activation::NONE,
            "ip2_val",
            val2);

    if (val2.size() != 3) {
        fail("value head output must be 3 logits");
    }
    softmax_inplace(val2.data(), 3);

    return WdlOutput{val2[0], val2[1], val2[2]};
}

}  // namespace makaira::lc0
