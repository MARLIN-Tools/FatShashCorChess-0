#pragma once

#include "lc0_activation.h"
#include "lc0_weights.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace makaira::lc0 {

class LinearBackend {
   public:
    enum class Type : int {
        SCALAR_FP32 = 0,
        INT8_KERNEL = 1,
        ONEDNN_FP32 = 2,
        ONEDNN_INT8 = 3,
        ORT_FP32 = 4,
        ORT_INT8 = 5,
    };

    void set_type(Type type);
    Type type() const { return type_; }
    void set_type_from_int(int type);
    int type_as_int() const { return static_cast<int>(type_); }
    void set_strict_fallback(bool strict) { strict_fallback_ = strict; }
    bool strict_fallback() const { return strict_fallback_; }

    const std::string& last_error() const { return last_error_; }
    std::string type_name() const;

    int infer_out_dim_no_bias(const Layer& w, int in_dim, const std::string& name) const;

    void fc_rows(const std::vector<float>& in,
                 int rows,
                 int in_dim,
                 const Layer& w,
                 const Layer& b,
                 Activation act,
                 const std::string& name,
                 std::vector<float>& out);

    void fc_rows_no_bias(const std::vector<float>& in,
                         int rows,
                         int in_dim,
                         const Layer& w,
                         Activation act,
                         const std::string& name,
                         std::vector<float>& out);

   private:
    struct QuantLayer {
        int in_dim = 0;
        int out_dim = 0;
        std::vector<std::int8_t> qweights{};
        std::vector<float> scales{};
    };

    struct Fp32Layer {
        int in_dim = 0;
        int out_dim = 0;
        std::vector<float> weights_transposed{};
    };

    const QuantLayer& get_quant_layer(const Layer& w, int in_dim, const std::string& name);
    const Fp32Layer& get_fp32_layer(const Layer& w, int in_dim, const std::string& name);
    void build_quant_layer(const Layer& w, int in_dim, const std::string& name, QuantLayer& out);
    void build_fp32_layer(const Layer& w, int in_dim, const std::string& name, Fp32Layer& out);
    float dot_int8(const std::int8_t* qw, const std::int8_t* qx, int n) const;

    void fc_rows_scalar(const std::vector<float>& in,
                        int rows,
                        int in_dim,
                        const Layer& w,
                        const Layer& b,
                        Activation act,
                        const std::string& name,
                        std::vector<float>& out) const;

    void fc_rows_no_bias_scalar(const std::vector<float>& in,
                                int rows,
                                int in_dim,
                                const Layer& w,
                                Activation act,
                                const std::string& name,
                                std::vector<float>& out) const;

    void fc_rows_int8(const std::vector<float>& in,
                      int rows,
                      int in_dim,
                      const Layer& w,
                      const Layer& b,
                      Activation act,
                      const std::string& name,
                      std::vector<float>& out);

    void fc_rows_no_bias_int8(const std::vector<float>& in,
                              int rows,
                              int in_dim,
                              const Layer& w,
                              Activation act,
                              const std::string& name,
                              std::vector<float>& out);

    void fc_rows_onednn(const std::vector<float>& in,
                        int rows,
                        int in_dim,
                        const Layer& w,
                        const Layer& b,
                        Activation act,
                        const std::string& name,
                        std::vector<float>& out);

    void fc_rows_no_bias_onednn(const std::vector<float>& in,
                                int rows,
                                int in_dim,
                                const Layer& w,
                                Activation act,
                                const std::string& name,
                                std::vector<float>& out);

    void fc_rows_ort(const std::vector<float>& in,
                     int rows,
                     int in_dim,
                     const Layer& w,
                     const Layer& b,
                     Activation act,
                     const std::string& name,
                     std::vector<float>& out);

    void fc_rows_no_bias_ort(const std::vector<float>& in,
                             int rows,
                             int in_dim,
                             const Layer& w,
                             Activation act,
                             const std::string& name,
                             std::vector<float>& out);

    Type type_ = Type::SCALAR_FP32;
    bool strict_fallback_ = false;
    std::string last_error_{};
    std::unordered_map<const Layer*, QuantLayer> quant_cache_{};
    std::unordered_map<const Layer*, Fp32Layer> fp32_cache_{};
    std::mutex quant_mutex_{};
    std::mutex fp32_mutex_{};
};

}  // namespace makaira::lc0
