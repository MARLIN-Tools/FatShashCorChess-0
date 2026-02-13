#include "lc0/lc0_weights.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>

int main() {
    const std::string path = "t1-256x10-distilled-swa-2432500.pb.gz";
    if (!std::filesystem::exists(path)) {
        std::cout << "[SKIP] lc0 weights file not found: " << path << "\n";
        return 0;
    }

    try {
        const auto w = makaira::lc0::load_from_pb_gz(path);
        makaira::lc0::validate_attention_value_shapes(w, true);

        if (static_cast<int>(w.encoders.size()) != 10) {
            std::cerr << "[FAIL] expected 10 encoders, got " << w.encoders.size() << "\n";
            return 1;
        }
        if (w.headcount != 8) {
            std::cerr << "[FAIL] expected headcount 8, got " << w.headcount << "\n";
            return 1;
        }
        if (w.ip_emb_b.values.size() != 256) {
            std::cerr << "[FAIL] expected embedding 256, got " << w.ip_emb_b.values.size() << "\n";
            return 1;
        }
        if (w.ip2_val_b.values.size() != 3) {
            std::cerr << "[FAIL] expected WDL output size 3, got " << w.ip2_val_b.values.size() << "\n";
            return 1;
        }

        const auto deq_ok = [](const makaira::lc0::Layer& layer) {
            if (layer.values.empty()) {
                return true;
            }
            if (layer.encoding != makaira::lc0::LayerEncoding::LINEAR16) {
                return true;
            }
            float min_seen = layer.values[0];
            float max_seen = layer.values[0];
            for (float v : layer.values) {
                min_seen = std::min(min_seen, v);
                max_seen = std::max(max_seen, v);
            }
            const float eps = std::max(1e-4f, std::abs(layer.max_val - layer.min_val) * 0.05f);
            return min_seen >= layer.min_val - eps && max_seen <= layer.max_val + eps;
        };

        if (!deq_ok(w.ip_emb_w)) {
            std::cerr << "[FAIL] dequantized ip_emb_w values exceed expected range\n";
            return 1;
        }

        std::cout << "[PASS] lc0 loader/shape/dequant checks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] lc0 loader exception: " << e.what() << "\n";
        return 1;
    }
}
