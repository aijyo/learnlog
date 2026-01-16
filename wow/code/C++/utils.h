#pragma once

#include <vector>

// -----------------------------
// Frame struct
// -----------------------------
struct BgraFrame
{
    int width = 0;
    int height = 0;
    int stride = 0; // bytes per row
    std::vector<uint8_t> data;
    uint64_t frame_id = 0;
};
