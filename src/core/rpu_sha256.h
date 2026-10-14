// SHA-256 for cold immutable payload identity; no runtime crypto dependency.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace rpu_payload_identity {
inline std::string sha256(std::string_view bytes) {
    constexpr std::array<uint32_t, 64> constants = {{
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    }};
    std::array<uint32_t, 8> state = {{
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    }};
    const auto rotate = [](uint32_t value, unsigned bits) {
        return (value >> bits) | (value << (32 - bits));
    };
    const auto compress = [&](const unsigned char* block) {
        std::array<uint32_t, 64> words{};
        for (size_t i = 0; i < 16; ++i) {
            const size_t j = 4 * i;
            words[i] = (uint32_t(block[j]) << 24) |
                (uint32_t(block[j + 1]) << 16) |
                (uint32_t(block[j + 2]) << 8) | uint32_t(block[j + 3]);
        }
        for (size_t i = 16; i < words.size(); ++i) {
            const uint32_t a = words[i - 15], b = words[i - 2];
            words[i] = words[i - 16] +
                (rotate(a, 7) ^ rotate(a, 18) ^ (a >> 3)) + words[i - 7] +
                (rotate(b, 17) ^ rotate(b, 19) ^ (b >> 10));
        }
        auto work = state;
        for (size_t i = 0; i < words.size(); ++i) {
            const uint32_t first = work[7] +
                (rotate(work[4], 6) ^ rotate(work[4], 11) ^ rotate(work[4], 25)) +
                ((work[4] & work[5]) ^ (~work[4] & work[6])) + constants[i] + words[i];
            const uint32_t second =
                (rotate(work[0], 2) ^ rotate(work[0], 13) ^ rotate(work[0], 22)) +
                ((work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2]));
            work = {{first + second, work[0], work[1], work[2],
                     work[3] + first, work[4], work[5], work[6]}};
        }
        for (size_t i = 0; i < state.size(); ++i) state[i] += work[i];
    };
    size_t offset = 0;
    while (bytes.size() - offset >= 64) {
        compress(reinterpret_cast<const unsigned char*>(bytes.data() + offset));
        offset += 64;
    }
    std::array<unsigned char, 128> tail{};
    const size_t remainder = bytes.size() - offset;
    for (size_t i = 0; i < remainder; ++i) tail[i] = static_cast<unsigned char>(bytes[offset + i]);
    tail[remainder] = 0x80;
    const size_t padded = remainder < 56 ? 64 : 128;
    const uint64_t bits = static_cast<uint64_t>(bytes.size()) * 8;
    for (size_t i = 0; i < 8; ++i) tail[padded - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
    compress(tail.data());
    if (padded == 128) compress(tail.data() + 64);
    constexpr char digits[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < state.size(); ++i) {
        for (size_t j = 0; j < 8; ++j) result[8 * i + j] = digits[(state[i] >> (28 - 4 * j)) & 15];
    }
    return result;
}
}  // namespace rpu_payload_identity
