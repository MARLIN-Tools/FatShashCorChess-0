#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace makaira::lc0 {

struct BytesView {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
};

enum class WireType : std::uint8_t {
    VARINT = 0,
    FIXED64 = 1,
    LENGTH_DELIMITED = 2,
    FIXED32 = 5,
};

struct FieldView {
    std::uint32_t number = 0;
    WireType wire_type = WireType::VARINT;
    std::uint64_t varint_value = 0;
    std::uint32_t fixed32_value = 0;
    std::uint64_t fixed64_value = 0;
    BytesView bytes{};
};

bool read_varint(BytesView view, std::size_t& offset, std::uint64_t& out);
bool next_field(BytesView view, std::size_t& offset, FieldView& out);

std::optional<FieldView> first_field(BytesView view, std::uint32_t field_number, WireType wire_type);
std::vector<FieldView> all_fields(BytesView view, std::uint32_t field_number, WireType wire_type);

std::optional<BytesView> first_submessage(BytesView view, std::uint32_t field_number);
std::vector<BytesView> all_submessages(BytesView view, std::uint32_t field_number);

std::vector<std::uint8_t> read_gzip_file(const std::string& path);

}  // namespace makaira::lc0
