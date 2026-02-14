#include "lc0_linear_backend.h"

#include "ort_gemm_models.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#if MAKAIRA_HAS_DNNL
#include <oneapi/dnnl/dnnl.hpp>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "../third_party/onnxruntime_c_api.h"

namespace makaira::lc0 {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("lc0 linear: " + msg);
}

int layer_output_size_local(const Layer& w, const Layer& b, const std::string& name) {
    const int out = static_cast<int>(b.values.size());
    if (out <= 0) {
        fail(name + ": bias vector is empty");
    }
    if (w.values.empty()) {
        fail(name + ": weight vector is empty");
    }
    if (static_cast<int>(w.values.size()) % out != 0) {
        fail(name + ": weight size not divisible by output size");
    }
    return out;
}

int layer_input_size_local(const Layer& w, const Layer& b, const std::string& name) {
    const int out = layer_output_size_local(w, b, name);
    return static_cast<int>(w.values.size()) / out;
}

#if defined(_WIN32)
std::string read_env_var(const char* name) {
    char* raw = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&raw, &len, name) != 0 || raw == nullptr) {
        return {};
    }
    std::string out(raw);
    std::free(raw);
    return out;
}
#endif

class OrtRuntime {
   public:
    static OrtRuntime& instance() {
        static OrtRuntime runtime;
        return runtime;
    }

    bool run_gemm(const std::vector<float>& in,
                  int rows,
                  int in_dim,
                  const std::vector<float>& w_row_major,
                  int out_dim,
                  const std::vector<float>* bias,
                  std::vector<float>& out,
                  std::string& err);

   private:
    struct SessionPair {
        OrtSession* bias = nullptr;
        OrtSession* nobias = nullptr;
        bool initialized = false;
    };

    OrtRuntime() = default;
    ~OrtRuntime();

    bool ensure_initialized(std::string& err);
    bool create_session_pair(SessionPair& pair, std::string& err);
    bool check_status(OrtStatus* st, const char* where, std::string& err) const;

    std::mutex mu_{};
    bool initialized_ = false;
    std::string init_error_{};

    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtMemoryInfo* cpu_mem_ = nullptr;
    OrtRunOptions* run_options_ = nullptr;
    SessionPair cpu_sessions_{};

#if defined(_WIN32)
    HMODULE dll_ = nullptr;
#endif
};

OrtRuntime::~OrtRuntime() {
    if (!api_) {
        return;
    }
    if (cpu_sessions_.bias) {
        api_->ReleaseSession(cpu_sessions_.bias);
    }
    if (cpu_sessions_.nobias) {
        api_->ReleaseSession(cpu_sessions_.nobias);
    }
    if (cpu_mem_) {
        api_->ReleaseMemoryInfo(cpu_mem_);
    }
    if (run_options_) {
        api_->ReleaseRunOptions(run_options_);
    }
    if (env_) {
        api_->ReleaseEnv(env_);
    }
#if defined(_WIN32)
    if (dll_) {
        FreeLibrary(dll_);
    }
#endif
}

bool OrtRuntime::check_status(OrtStatus* st, const char* where, std::string& err) const {
    if (!st) {
        return true;
    }
    const char* msg = api_ ? api_->GetErrorMessage(st) : "unknown ORT error";
    err = std::string(where) + ": " + (msg ? msg : "unknown ORT error");
    if (api_) {
        api_->ReleaseStatus(st);
    }
    return false;
}

bool OrtRuntime::ensure_initialized(std::string& err) {
    std::scoped_lock lock(mu_);
    if (initialized_) {
        if (!init_error_.empty()) {
            err = init_error_;
            return false;
        }
        return true;
    }
    initialized_ = true;

#if defined(_WIN32)
    std::vector<std::string> candidates{};
    const std::string env_path = read_env_var("MAKAIRA_ORT_DLL");
    if (!env_path.empty()) {
        candidates.emplace_back(env_path);
    }
    candidates.emplace_back("onnxruntime.dll");
    const std::string local = read_env_var("LOCALAPPDATA");
    if (!local.empty()) {
        for (const char* ver : {"39", "310", "311", "312", "313"}) {
            candidates.emplace_back(local + "\\Programs\\Python\\Python" + ver
                                    + "\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime.dll");
        }
    }
    for (const auto& c : candidates) {
        dll_ = LoadLibraryA(c.c_str());
        if (dll_) {
            break;
        }
    }
    if (!dll_) {
        init_error_ = "failed to load onnxruntime.dll (set MAKAIRA_ORT_DLL)";
        err = init_error_;
        return false;
    }
    auto get_api_base = reinterpret_cast<const OrtApiBase*(ORT_API_CALL*)(void)>(GetProcAddress(dll_, "OrtGetApiBase"));
    if (!get_api_base) {
        init_error_ = "onnxruntime.dll missing OrtGetApiBase";
        err = init_error_;
        return false;
    }
    const OrtApiBase* base = get_api_base();
    if (!base) {
        init_error_ = "OrtGetApiBase returned null";
        err = init_error_;
        return false;
    }
    // Header and DLL may differ. ORT 1.17 is common locally, so probe that range first.
    const uint32_t start_version = ORT_API_VERSION > 17 ? 17 : ORT_API_VERSION;
    for (uint32_t v = start_version; v >= 1; --v) {
        api_ = base->GetApi(v);
        if (api_) {
            break;
        }
    }
    if (!api_) {
        init_error_ = "failed to negotiate an ONNX Runtime API version";
        err = init_error_;
        return false;
    }
#else
    init_error_ = "ORT runtime loading is only implemented on Windows";
    err = init_error_;
    return false;
#endif

    OrtStatus* st = api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "makaira", &env_);
    if (!check_status(st, "CreateEnv", init_error_)) {
        err = init_error_;
        return false;
    }
    st = api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_mem_);
    if (!check_status(st, "CreateCpuMemoryInfo", init_error_)) {
        err = init_error_;
        return false;
    }
    st = api_->CreateRunOptions(&run_options_);
    if (!check_status(st, "CreateRunOptions", init_error_)) {
        err = init_error_;
        return false;
    }

    err.clear();
    return true;
}

bool OrtRuntime::create_session_pair(SessionPair& pair, std::string& err) {
    OrtSessionOptions* so = nullptr;
    OrtStatus* st = api_->CreateSessionOptions(&so);
    if (!check_status(st, "CreateSessionOptions", err)) {
        return false;
    }
    st = api_->SetIntraOpNumThreads(so, 1);
    if (!check_status(st, "SetIntraOpNumThreads", err)) {
        api_->ReleaseSessionOptions(so);
        return false;
    }
    st = api_->SetInterOpNumThreads(so, 1);
    if (!check_status(st, "SetInterOpNumThreads", err)) {
        api_->ReleaseSessionOptions(so);
        return false;
    }
    st = api_->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_EXTENDED);
    if (!check_status(st, "SetSessionGraphOptimizationLevel", err)) {
        api_->ReleaseSessionOptions(so);
        return false;
    }
    st = api_->CreateSessionFromArray(env_, kOrtGemmBiasModel, kOrtGemmBiasModel_size, so, &pair.bias);
    if (!check_status(st, "CreateSessionFromArray(gemm_bias)", err)) {
        api_->ReleaseSessionOptions(so);
        return false;
    }
    st = api_->CreateSessionFromArray(env_, kOrtGemmNoBiasModel, kOrtGemmNoBiasModel_size, so, &pair.nobias);
    if (!check_status(st, "CreateSessionFromArray(gemm_nobias)", err)) {
        api_->ReleaseSession(pair.bias);
        pair.bias = nullptr;
        api_->ReleaseSessionOptions(so);
        return false;
    }
    api_->ReleaseSessionOptions(so);
    pair.initialized = true;
    return true;
}

bool OrtRuntime::run_gemm(const std::vector<float>& in,
                          int rows,
                          int in_dim,
                          const std::vector<float>& w_row_major,
                          int out_dim,
                          const std::vector<float>* bias,
                          std::vector<float>& out,
                          std::string& err) {
    if (!ensure_initialized(err)) {
        return false;
    }
    if (!cpu_sessions_.initialized && !create_session_pair(cpu_sessions_, err)) {
        return false;
    }

    OrtSession* session = bias ? cpu_sessions_.bias : cpu_sessions_.nobias;
    if (!session) {
        err = "ORT session is null";
        return false;
    }
    if (rows <= 0 || in_dim <= 0 || out_dim <= 0) {
        err = "invalid GEMM dimensions";
        return false;
    }
    if (static_cast<int>(in.size()) != rows * in_dim) {
        err = "input size mismatch for ORT GEMM";
        return false;
    }
    if (static_cast<int>(w_row_major.size()) != out_dim * in_dim) {
        err = "weight size mismatch for ORT GEMM";
        return false;
    }
    if (bias && static_cast<int>(bias->size()) != out_dim) {
        err = "bias size mismatch for ORT GEMM";
        return false;
    }

    out.resize(static_cast<std::size_t>(rows * out_dim));

    const int64_t x_dims[2] = {rows, in_dim};
    const int64_t w_dims[2] = {out_dim, in_dim};  // model uses transB=1
    const int64_t y_dims[2] = {rows, out_dim};
    const int64_t b_dims[1] = {out_dim};

    OrtValue* x_val = nullptr;
    OrtValue* w_val = nullptr;
    OrtValue* b_val = nullptr;
    OrtValue* y_val = nullptr;

    auto release_value = [this](OrtValue*& v) {
        if (v) {
            api_->ReleaseValue(v);
            v = nullptr;
        }
    };

    OrtStatus* st = api_->CreateTensorWithDataAsOrtValue(cpu_mem_,
                                                          const_cast<float*>(in.data()),
                                                          in.size() * sizeof(float),
                                                          x_dims,
                                                          2,
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                          &x_val);
    if (!check_status(st, "CreateTensor(X)", err)) {
        release_value(x_val);
        return false;
    }

    st = api_->CreateTensorWithDataAsOrtValue(cpu_mem_,
                                              const_cast<float*>(w_row_major.data()),
                                              w_row_major.size() * sizeof(float),
                                              w_dims,
                                              2,
                                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                              &w_val);
    if (!check_status(st, "CreateTensor(W)", err)) {
        release_value(x_val);
        release_value(w_val);
        return false;
    }

    if (bias) {
        st = api_->CreateTensorWithDataAsOrtValue(cpu_mem_,
                                                  const_cast<float*>(bias->data()),
                                                  bias->size() * sizeof(float),
                                                  b_dims,
                                                  1,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                  &b_val);
        if (!check_status(st, "CreateTensor(B)", err)) {
            release_value(x_val);
            release_value(w_val);
            release_value(b_val);
            return false;
        }
    }

    st = api_->CreateTensorWithDataAsOrtValue(cpu_mem_,
                                              out.data(),
                                              out.size() * sizeof(float),
                                              y_dims,
                                              2,
                                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                              &y_val);
    if (!check_status(st, "CreateTensor(Y)", err)) {
        release_value(x_val);
        release_value(w_val);
        release_value(b_val);
        release_value(y_val);
        return false;
    }

    static constexpr const char* kInputNamesBias[] = {"X", "W", "B"};
    static constexpr const char* kInputNamesNoBias[] = {"X", "W"};
    static constexpr const char* kOutputNames[] = {"Y"};

    std::array<const OrtValue*, 3> inputs_bias{x_val, w_val, b_val};
    std::array<const OrtValue*, 2> inputs_nobias{x_val, w_val};
    std::array<OrtValue*, 1> outputs{y_val};

    st = api_->Run(session,
                   run_options_,
                   bias ? kInputNamesBias : kInputNamesNoBias,
                   bias ? inputs_bias.data() : inputs_nobias.data(),
                   bias ? 3u : 2u,
                   kOutputNames,
                   1u,
                   outputs.data());

    release_value(x_val);
    release_value(w_val);
    release_value(b_val);
    release_value(y_val);

    return check_status(st, "Run(Gemm)", err);
}

}  // namespace

void LinearBackend::set_type(Type type) {
    type_ = type;
}

void LinearBackend::set_type_from_int(int type) {
    if (type <= 0) {
        set_type(Type::SCALAR_FP32);
    } else if (type == 1) {
        set_type(Type::INT8_KERNEL);
    } else if (type == 2) {
        set_type(Type::ONEDNN_FP32);
    } else if (type == 3) {
        set_type(Type::ONEDNN_INT8);
    } else if (type == 4) {
        set_type(Type::ORT_FP32);
    } else {
        set_type(Type::ORT_INT8);
    }
}

std::string LinearBackend::type_name() const {
    switch (type_) {
        case Type::SCALAR_FP32:
            return "scalar_fp32";
        case Type::INT8_KERNEL:
            return "int8_kernel";
        case Type::ONEDNN_FP32:
            return "onednn_fp32";
        case Type::ONEDNN_INT8:
            return "onednn_int8";
        case Type::ORT_FP32:
            return "ort_fp32";
        case Type::ORT_INT8:
            return "ort_int8";
        default:
            return "unknown";
    }
}

int LinearBackend::infer_out_dim_no_bias(const Layer& w, int in_dim, const std::string& name) const {
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

void LinearBackend::fc_rows(const std::vector<float>& in,
                            int rows,
                            int in_dim,
                            const Layer& w,
                            const Layer& b,
                            Activation act,
                            const std::string& name,
                            std::vector<float>& out) {
    last_error_.clear();

    if (type_ == Type::SCALAR_FP32) {
        fc_rows_scalar(in, rows, in_dim, w, b, act, name, out);
        return;
    }
    if (type_ == Type::INT8_KERNEL || type_ == Type::ORT_INT8) {
        fc_rows_int8(in, rows, in_dim, w, b, act, name, out);
        return;
    }
    if (type_ == Type::ONEDNN_FP32 || type_ == Type::ONEDNN_INT8) {
        fc_rows_onednn(in, rows, in_dim, w, b, act, name, out);
        return;
    }
    if (type_ == Type::ORT_FP32) {
        fc_rows_ort(in, rows, in_dim, w, b, act, name, out);
        return;
    }
    fc_rows_scalar(in, rows, in_dim, w, b, act, name, out);
}

void LinearBackend::fc_rows_no_bias(const std::vector<float>& in,
                                    int rows,
                                    int in_dim,
                                    const Layer& w,
                                    Activation act,
                                    const std::string& name,
                                    std::vector<float>& out) {
    last_error_.clear();

    if (type_ == Type::SCALAR_FP32) {
        fc_rows_no_bias_scalar(in, rows, in_dim, w, act, name, out);
        return;
    }
    if (type_ == Type::INT8_KERNEL || type_ == Type::ORT_INT8) {
        fc_rows_no_bias_int8(in, rows, in_dim, w, act, name, out);
        return;
    }
    if (type_ == Type::ONEDNN_FP32 || type_ == Type::ONEDNN_INT8) {
        fc_rows_no_bias_onednn(in, rows, in_dim, w, act, name, out);
        return;
    }
    if (type_ == Type::ORT_FP32) {
        fc_rows_no_bias_ort(in, rows, in_dim, w, act, name, out);
        return;
    }
    fc_rows_no_bias_scalar(in, rows, in_dim, w, act, name, out);
}

void LinearBackend::fc_rows_scalar(const std::vector<float>& in,
                                   int rows,
                                   int in_dim,
                                   const Layer& w,
                                   const Layer& b,
                                   Activation act,
                                   const std::string& name,
                                   std::vector<float>& out) const {
    const int out_dim = layer_output_size_local(w, b, name);
    const int inferred_in = layer_input_size_local(w, b, name);
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

void LinearBackend::fc_rows_no_bias_scalar(const std::vector<float>& in,
                                           int rows,
                                           int in_dim,
                                           const Layer& w,
                                           Activation act,
                                           const std::string& name,
                                           std::vector<float>& out) const {
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

const LinearBackend::QuantLayer& LinearBackend::get_quant_layer(const Layer& w, int in_dim, const std::string& name) {
    std::scoped_lock lock(quant_mutex_);
    auto it = quant_cache_.find(&w);
    if (it != quant_cache_.end()) {
        return it->second;
    }
    QuantLayer q{};
    build_quant_layer(w, in_dim, name, q);
    auto [new_it, ok] = quant_cache_.emplace(&w, std::move(q));
    (void)ok;
    return new_it->second;
}

void LinearBackend::build_quant_layer(const Layer& w, int in_dim, const std::string& name, QuantLayer& out) {
    const int out_dim = infer_out_dim_no_bias(w, in_dim, name);
    out.in_dim = in_dim;
    out.out_dim = out_dim;
    out.qweights.assign(static_cast<std::size_t>(in_dim * out_dim), 0);
    out.scales.assign(static_cast<std::size_t>(out_dim), 1.0f);

    for (int o = 0; o < out_dim; ++o) {
        const float* ww = w.values.data() + static_cast<std::size_t>(o * in_dim);
        float max_abs = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            max_abs = std::max(max_abs, std::abs(ww[i]));
        }
        const float scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
        out.scales[static_cast<std::size_t>(o)] = scale;
        const float inv = scale > 0.0f ? (1.0f / scale) : 0.0f;
        std::int8_t* qw = out.qweights.data() + static_cast<std::size_t>(o * in_dim);
        for (int i = 0; i < in_dim; ++i) {
            const int q = static_cast<int>(std::lround(ww[i] * inv));
            qw[i] = static_cast<std::int8_t>(std::clamp(q, -127, 127));
        }
    }
}

const LinearBackend::Fp32Layer& LinearBackend::get_fp32_layer(const Layer& w, int in_dim, const std::string& name) {
    std::scoped_lock lock(fp32_mutex_);
    auto it = fp32_cache_.find(&w);
    if (it != fp32_cache_.end()) {
        return it->second;
    }
    Fp32Layer fp{};
    build_fp32_layer(w, in_dim, name, fp);
    auto [new_it, ok] = fp32_cache_.emplace(&w, std::move(fp));
    (void)ok;
    return new_it->second;
}

void LinearBackend::build_fp32_layer(const Layer& w, int in_dim, const std::string& name, Fp32Layer& out) {
    const int out_dim = infer_out_dim_no_bias(w, in_dim, name);
    out.in_dim = in_dim;
    out.out_dim = out_dim;
    out.weights_transposed.assign(static_cast<std::size_t>(in_dim * out_dim), 0.0f);
    for (int o = 0; o < out_dim; ++o) {
        for (int i = 0; i < in_dim; ++i) {
            out.weights_transposed[static_cast<std::size_t>(i * out_dim + o)] =
              w.values[static_cast<std::size_t>(o * in_dim + i)];
        }
    }
}

float LinearBackend::dot_int8(const std::int8_t* qw, const std::int8_t* qx, int n) const {
    std::int32_t acc = 0;
    for (int i = 0; i < n; ++i) {
        acc += static_cast<std::int32_t>(qw[i]) * static_cast<std::int32_t>(qx[i]);
    }
    return static_cast<float>(acc);
}

void LinearBackend::fc_rows_int8(const std::vector<float>& in,
                                 int rows,
                                 int in_dim,
                                 const Layer& w,
                                 const Layer& b,
                                 Activation act,
                                 const std::string& name,
                                 std::vector<float>& out) {
    const int out_dim = layer_output_size_local(w, b, name);
    if (in_dim != layer_input_size_local(w, b, name)) {
        fail(name + ": in_dim mismatch");
    }
    if (static_cast<int>(in.size()) != rows * in_dim) {
        fail(name + ": input vector size mismatch");
    }

    const QuantLayer& qw = get_quant_layer(w, in_dim, name);
    out.assign(static_cast<std::size_t>(rows * out_dim), 0.0f);
    std::vector<std::int8_t> qx(static_cast<std::size_t>(in_dim), 0);

    for (int r = 0; r < rows; ++r) {
        const float* x = in.data() + static_cast<std::size_t>(r * in_dim);
        float max_abs = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            max_abs = std::max(max_abs, std::abs(x[i]));
        }
        const float sx = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
        const float inv_sx = sx > 0.0f ? (1.0f / sx) : 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            const int q = static_cast<int>(std::lround(x[i] * inv_sx));
            qx[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(std::clamp(q, -127, 127));
        }

        float* y = out.data() + static_cast<std::size_t>(r * out_dim);
        for (int o = 0; o < out_dim; ++o) {
            const std::int8_t* wrow = qw.qweights.data() + static_cast<std::size_t>(o * in_dim);
            const float acc = dot_int8(wrow, qx.data(), in_dim);
            float sum = b.values[static_cast<std::size_t>(o)] + acc * sx * qw.scales[static_cast<std::size_t>(o)];
            y[o] = activate_scalar(sum, act);
        }
    }
}

void LinearBackend::fc_rows_no_bias_int8(const std::vector<float>& in,
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

    const QuantLayer& qw = get_quant_layer(w, in_dim, name);
    out.assign(static_cast<std::size_t>(rows * out_dim), 0.0f);
    std::vector<std::int8_t> qx(static_cast<std::size_t>(in_dim), 0);

    for (int r = 0; r < rows; ++r) {
        const float* x = in.data() + static_cast<std::size_t>(r * in_dim);
        float max_abs = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            max_abs = std::max(max_abs, std::abs(x[i]));
        }
        const float sx = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
        const float inv_sx = sx > 0.0f ? (1.0f / sx) : 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            const int q = static_cast<int>(std::lround(x[i] * inv_sx));
            qx[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(std::clamp(q, -127, 127));
        }

        float* y = out.data() + static_cast<std::size_t>(r * out_dim);
        for (int o = 0; o < out_dim; ++o) {
            const std::int8_t* wrow = qw.qweights.data() + static_cast<std::size_t>(o * in_dim);
            const float acc = dot_int8(wrow, qx.data(), in_dim);
            float sum = acc * sx * qw.scales[static_cast<std::size_t>(o)];
            y[o] = activate_scalar(sum, act);
        }
    }
}

void LinearBackend::fc_rows_onednn(const std::vector<float>& in,
                                   int rows,
                                   int in_dim,
                                   const Layer& w,
                                   const Layer& b,
                                   Activation act,
                                   const std::string& name,
                                   std::vector<float>& out) {
#if MAKAIRA_HAS_DNNL
    const int out_dim = layer_output_size_local(w, b, name);
    if (in_dim != layer_input_size_local(w, b, name)) {
        fail(name + ": in_dim mismatch");
    }
    if (static_cast<int>(in.size()) != rows * in_dim) {
        fail(name + ": input vector size mismatch");
    }

    const Fp32Layer& fp = get_fp32_layer(w, in_dim, name);
    out.assign(static_cast<std::size_t>(rows * out_dim), 0.0f);

    static oneapi::dnnl::engine eng(oneapi::dnnl::engine::kind::cpu, 0);
    thread_local oneapi::dnnl::stream strm(eng);

    auto src_md = oneapi::dnnl::memory::desc({rows, in_dim}, oneapi::dnnl::memory::data_type::f32, oneapi::dnnl::memory::format_tag::ab);
    auto wei_md = oneapi::dnnl::memory::desc({in_dim, out_dim}, oneapi::dnnl::memory::data_type::f32, oneapi::dnnl::memory::format_tag::ab);
    auto dst_md = oneapi::dnnl::memory::desc({rows, out_dim}, oneapi::dnnl::memory::data_type::f32, oneapi::dnnl::memory::format_tag::ab);
    auto src_mem = oneapi::dnnl::memory(src_md, eng, const_cast<float*>(in.data()));
    auto wei_mem = oneapi::dnnl::memory(wei_md, eng, const_cast<float*>(fp.weights_transposed.data()));
    auto dst_mem = oneapi::dnnl::memory(dst_md, eng, out.data());
    auto pd = oneapi::dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md);
    auto prim = oneapi::dnnl::matmul(pd);
    prim.execute(strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem}, {DNNL_ARG_DST, dst_mem}});
    strm.wait();

    for (int r = 0; r < rows; ++r) {
        float* y = out.data() + static_cast<std::size_t>(r * out_dim);
        for (int o = 0; o < out_dim; ++o) {
            y[o] = activate_scalar(y[o] + b.values[static_cast<std::size_t>(o)], act);
        }
    }
#else
    last_error_ = "oneDNN backend requested but build has no oneDNN; using scalar fallback";
    fc_rows_scalar(in, rows, in_dim, w, b, act, name, out);
#endif
}

void LinearBackend::fc_rows_no_bias_onednn(const std::vector<float>& in,
                                           int rows,
                                           int in_dim,
                                           const Layer& w,
                                           Activation act,
                                           const std::string& name,
                                           std::vector<float>& out) {
#if MAKAIRA_HAS_DNNL
    const int out_dim = infer_out_dim_no_bias(w, in_dim, name);
    if (static_cast<int>(in.size()) != rows * in_dim) {
        fail(name + ": input vector size mismatch");
    }

    const Fp32Layer& fp = get_fp32_layer(w, in_dim, name);
    out.assign(static_cast<std::size_t>(rows * out_dim), 0.0f);

    static oneapi::dnnl::engine eng(oneapi::dnnl::engine::kind::cpu, 0);
    thread_local oneapi::dnnl::stream strm(eng);

    auto src_md = oneapi::dnnl::memory::desc({rows, in_dim}, oneapi::dnnl::memory::data_type::f32, oneapi::dnnl::memory::format_tag::ab);
    auto wei_md = oneapi::dnnl::memory::desc({in_dim, out_dim}, oneapi::dnnl::memory::data_type::f32, oneapi::dnnl::memory::format_tag::ab);
    auto dst_md = oneapi::dnnl::memory::desc({rows, out_dim}, oneapi::dnnl::memory::data_type::f32, oneapi::dnnl::memory::format_tag::ab);
    auto src_mem = oneapi::dnnl::memory(src_md, eng, const_cast<float*>(in.data()));
    auto wei_mem = oneapi::dnnl::memory(wei_md, eng, const_cast<float*>(fp.weights_transposed.data()));
    auto dst_mem = oneapi::dnnl::memory(dst_md, eng, out.data());
    auto pd = oneapi::dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md);
    auto prim = oneapi::dnnl::matmul(pd);
    prim.execute(strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem}, {DNNL_ARG_DST, dst_mem}});
    strm.wait();

    for (float& v : out) {
        v = activate_scalar(v, act);
    }
#else
    last_error_ = "oneDNN backend requested but build has no oneDNN; using scalar fallback";
    fc_rows_no_bias_scalar(in, rows, in_dim, w, act, name, out);
#endif
}

void LinearBackend::fc_rows_ort(const std::vector<float>& in,
                                int rows,
                                int in_dim,
                                const Layer& w,
                                const Layer& b,
                                Activation act,
                                const std::string& name,
                                std::vector<float>& out) {
    const int out_dim = layer_output_size_local(w, b, name);
    const int inferred_in = layer_input_size_local(w, b, name);
    if (inferred_in != in_dim) {
        fail(name + ": expected in_dim " + std::to_string(inferred_in) + ", got " + std::to_string(in_dim));
    }

    std::string err{};
    const bool ok = OrtRuntime::instance().run_gemm(in, rows, in_dim, w.values, out_dim, &b.values, out, err);
    if (!ok) {
        last_error_ = "ORT backend failure: " + err + "; using scalar fallback";
        fc_rows_scalar(in, rows, in_dim, w, b, act, name, out);
        return;
    }
    for (float& v : out) {
        v = activate_scalar(v, act);
    }
}

void LinearBackend::fc_rows_no_bias_ort(const std::vector<float>& in,
                                        int rows,
                                        int in_dim,
                                        const Layer& w,
                                        Activation act,
                                        const std::string& name,
                                        std::vector<float>& out) {
    const int out_dim = infer_out_dim_no_bias(w, in_dim, name);
    std::string err{};
    const bool ok = OrtRuntime::instance().run_gemm(in, rows, in_dim, w.values, out_dim, nullptr, out, err);
    if (!ok) {
        last_error_ = "ORT backend failure: " + err + "; using scalar fallback";
        fc_rows_no_bias_scalar(in, rows, in_dim, w, act, name, out);
        return;
    }
    for (float& v : out) {
        v = activate_scalar(v, act);
    }
}

}  // namespace makaira::lc0
