#include "lc0_proto_wire.h"

#include <cstring>
#include <stdexcept>

#if MAKAIRA_HAS_ZLIB
#include <zlib.h>
#endif

#if !MAKAIRA_HAS_ZLIB && defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace makaira::lc0 {

namespace {

std::uint32_t read_u32_le(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0])
         | (static_cast<std::uint32_t>(p[1]) << 8)
         | (static_cast<std::uint32_t>(p[2]) << 16)
         | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t read_u64_le(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

}  // namespace

bool read_varint(BytesView view, std::size_t& offset, std::uint64_t& out) {
    out = 0;
    int shift = 0;
    while (offset < view.size) {
        const std::uint8_t b = view.data[offset++];
        out |= (static_cast<std::uint64_t>(b & 0x7Fu) << shift);
        if ((b & 0x80u) == 0u) {
            return true;
        }
        shift += 7;
        if (shift > 63) {
            return false;
        }
    }
    return false;
}

bool next_field(BytesView view, std::size_t& offset, FieldView& out) {
    if (offset >= view.size) {
        return false;
    }

    std::uint64_t key = 0;
    if (!read_varint(view, offset, key)) {
        throw std::runtime_error("protobuf parse error: invalid field key varint");
    }

    out = FieldView{};
    out.number = static_cast<std::uint32_t>(key >> 3);
    const std::uint8_t wt = static_cast<std::uint8_t>(key & 7u);

    switch (wt) {
        case 0: {
            out.wire_type = WireType::VARINT;
            if (!read_varint(view, offset, out.varint_value)) {
                throw std::runtime_error("protobuf parse error: invalid varint value");
            }
            break;
        }
        case 1: {
            out.wire_type = WireType::FIXED64;
            if (offset + 8 > view.size) {
                throw std::runtime_error("protobuf parse error: truncated fixed64 field");
            }
            out.fixed64_value = read_u64_le(view.data + offset);
            offset += 8;
            break;
        }
        case 2: {
            out.wire_type = WireType::LENGTH_DELIMITED;
            std::uint64_t len = 0;
            if (!read_varint(view, offset, len)) {
                throw std::runtime_error("protobuf parse error: invalid length-delimited size");
            }
            if (len > static_cast<std::uint64_t>(view.size - offset)) {
                throw std::runtime_error("protobuf parse error: truncated length-delimited field");
            }
            out.bytes = BytesView{view.data + offset, static_cast<std::size_t>(len)};
            offset += static_cast<std::size_t>(len);
            break;
        }
        case 5: {
            out.wire_type = WireType::FIXED32;
            if (offset + 4 > view.size) {
                throw std::runtime_error("protobuf parse error: truncated fixed32 field");
            }
            out.fixed32_value = read_u32_le(view.data + offset);
            offset += 4;
            break;
        }
        default:
            throw std::runtime_error("protobuf parse error: unsupported wire type " + std::to_string(wt));
    }

    return true;
}

std::optional<FieldView> first_field(BytesView view, std::uint32_t field_number, WireType wire_type) {
    std::size_t offset = 0;
    FieldView f;
    while (next_field(view, offset, f)) {
        if (f.number == field_number && f.wire_type == wire_type) {
            return f;
        }
    }
    return std::nullopt;
}

std::vector<FieldView> all_fields(BytesView view, std::uint32_t field_number, WireType wire_type) {
    std::vector<FieldView> out;
    std::size_t offset = 0;
    FieldView f;
    while (next_field(view, offset, f)) {
        if (f.number == field_number && f.wire_type == wire_type) {
            out.push_back(f);
        }
    }
    return out;
}

std::optional<BytesView> first_submessage(BytesView view, std::uint32_t field_number) {
    const auto f = first_field(view, field_number, WireType::LENGTH_DELIMITED);
    if (!f.has_value()) {
        return std::nullopt;
    }
    return f->bytes;
}

std::vector<BytesView> all_submessages(BytesView view, std::uint32_t field_number) {
    std::vector<BytesView> out;
    const auto fields = all_fields(view, field_number, WireType::LENGTH_DELIMITED);
    out.reserve(fields.size());
    for (const auto& f : fields) {
        out.push_back(f.bytes);
    }
    return out;
}

std::vector<std::uint8_t> read_gzip_file(const std::string& path) {
#if MAKAIRA_HAS_ZLIB
    gzFile file = gzopen(path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("lc0 loader: cannot open gzip file: " + path);
    }

    std::vector<std::uint8_t> buffer;
    buffer.resize(1 << 20);
    std::size_t used = 0;

    while (true) {
        if (used == buffer.size()) {
            buffer.resize(buffer.size() * 2);
        }

        const int got = gzread(file, buffer.data() + used, static_cast<unsigned int>(buffer.size() - used));
        if (got < 0) {
            int errnum = 0;
            const char* msg = gzerror(file, &errnum);
            gzclose(file);
            throw std::runtime_error(std::string("lc0 loader: gzread failed: ") + (msg ? msg : "unknown"));
        }
        if (got == 0) {
            break;
        }
        used += static_cast<std::size_t>(got);
    }

    gzclose(file);
    buffer.resize(used);
    return buffer;
#elif defined(_WIN32)
    using gzFile = void*;
    using gzopen_fn = gzFile(__cdecl*)(const char*, const char*);
    using gzread_fn = int(__cdecl*)(gzFile, void*, unsigned int);
    using gzclose_fn = int(__cdecl*)(gzFile);
    using gzerror_fn = const char*(__cdecl*)(gzFile, int*);

    HMODULE zlib_mod = LoadLibraryA("zlib1.dll");
    if (!zlib_mod) {
        throw std::runtime_error("lc0 loader: zlib support unavailable (zlib1.dll not found)");
    }

    const auto p_gzopen = reinterpret_cast<gzopen_fn>(GetProcAddress(zlib_mod, "gzopen"));
    const auto p_gzread = reinterpret_cast<gzread_fn>(GetProcAddress(zlib_mod, "gzread"));
    const auto p_gzclose = reinterpret_cast<gzclose_fn>(GetProcAddress(zlib_mod, "gzclose"));
    const auto p_gzerror = reinterpret_cast<gzerror_fn>(GetProcAddress(zlib_mod, "gzerror"));
    if (!p_gzopen || !p_gzread || !p_gzclose || !p_gzerror) {
        FreeLibrary(zlib_mod);
        throw std::runtime_error("lc0 loader: zlib1.dll missing gz* symbols");
    }

    gzFile file = p_gzopen(path.c_str(), "rb");
    if (!file) {
        FreeLibrary(zlib_mod);
        throw std::runtime_error("lc0 loader: cannot open gzip file: " + path);
    }

    std::vector<std::uint8_t> buffer;
    buffer.resize(1 << 20);
    std::size_t used = 0;

    while (true) {
        if (used == buffer.size()) {
            buffer.resize(buffer.size() * 2);
        }

        const int got = p_gzread(file, buffer.data() + used, static_cast<unsigned int>(buffer.size() - used));
        if (got < 0) {
            int errnum = 0;
            const char* msg = p_gzerror(file, &errnum);
            p_gzclose(file);
            FreeLibrary(zlib_mod);
            throw std::runtime_error(std::string("lc0 loader: gzread failed: ") + (msg ? msg : "unknown"));
        }
        if (got == 0) {
            break;
        }
        used += static_cast<std::size_t>(got);
    }

    p_gzclose(file);
    FreeLibrary(zlib_mod);
    buffer.resize(used);
    return buffer;
#else
    (void)path;
    throw std::runtime_error("lc0 loader: zlib support is disabled at build time (MAKAIRA_HAS_ZLIB=0)");
#endif
}

}  // namespace makaira::lc0
