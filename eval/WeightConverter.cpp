#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <array>
#include <memory>

using namespace std;

constexpr int FeatureDim = 64;
constexpr int PolicyDim = 32;
constexpr int ValueDim = 64;
constexpr int FeatDWConvDim = 32;
constexpr int PolicyPWConvDim = 16;
constexpr int NumHeadBucket = 1;
constexpr int ShapeNum = 442503;

template <int OutSize, int InSize>
struct FCWeight
{
    int8_t  weight[OutSize * InSize];
    int32_t bias[OutSize];
};

template <int OutSize, int InSize>
struct StarBlockWeight
{
    FCWeight<OutSize * 2, InSize> value_corner_up1;
    FCWeight<OutSize * 2, InSize> value_corner_up2;
    FCWeight<OutSize, OutSize>    value_corner_down;
};

// OLD Layout
struct alignas(64) OldWeight
{
    // 1  mapping layer
    int16_t  codebook[2][65536][FeatureDim];
    uint16_t mapping_index[2][ShapeNum];
    char     __padding_to_64bytes_0[36];

    // 2  Depthwise conv
    int16_t feature_dwconv_weight[9][FeatDWConvDim];
    int16_t feature_dwconv_bias[FeatDWConvDim];

    struct HeadBucket
    {
        // 3  Policy dynamic pointwise conv
        FCWeight<PolicyDim * 2, FeatureDim> policy_pwconv_layer_l1;
        FCWeight<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2>
            policy_pwconv_layer_l2;

        // 4  Value Group MLP (layer 1,2)
        StarBlockWeight<ValueDim, FeatureDim> value_corner;
        StarBlockWeight<ValueDim, FeatureDim> value_edge;
        StarBlockWeight<ValueDim, FeatureDim> value_center;
        StarBlockWeight<ValueDim, ValueDim>   value_quad;

        // 5  Value MLP (layer 1,2,3)
        FCWeight<ValueDim, FeatureDim + ValueDim * 4> value_l1;
        FCWeight<ValueDim, ValueDim>                  value_l2;
        FCWeight<4, ValueDim>                         value_l3;

        // 6  Policy output linear
        float policy_output_weight[16];
        float policy_output_bias;
        char  __padding_to_64bytes_1[44];
    } buckets[NumHeadBucket];
};

// NEW Layout
struct alignas(64) NewWeight
{
    // 1  mapping layer
    int16_t  codebook[2][65536][FeatureDim];
    uint16_t mapping_index[2][ShapeNum];
    char     __padding_to_64bytes_0[36];

    // 2  Depthwise conv
    int16_t feature_dwconv_weight[9][FeatDWConvDim];
    int16_t feature_dwconv_bias[FeatDWConvDim];

    struct HeadBucket
    {
        // 3  Policy dynamic pointwise conv
        FCWeight<PolicyDim * 2, FeatureDim> policy_pwconv_layer_l1;
        FCWeight<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2>
            policy_pwconv_layer_l2;

        // 4  Value Group MLP (layer 1,2)
        StarBlockWeight<ValueDim, FeatureDim> value_corner;
        StarBlockWeight<ValueDim, FeatureDim> value_edge;
        StarBlockWeight<ValueDim, FeatureDim> value_center;
        StarBlockWeight<ValueDim, ValueDim>   value_quad;

        // 5  Value MLP (layer 1,2,3)
        FCWeight<ValueDim, FeatureDim + ValueDim * 4> value_l1;
        FCWeight<ValueDim, ValueDim>                  value_l2;
        FCWeight<4, ValueDim>                         value_l3;

        // 6  Policy output linear
        alignas(32) float policy_output_weight[16];
        float policy_output_bias;
        char  __padding_to_64bytes_1[28]; // Adjusted padding
    } buckets[NumHeadBucket];
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_old_weight> <output_new_weight>" << std::endl;
        return 1;
    }

    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input file: " << argv[1] << std::endl;
        return 1;
    }

    // Allocate on heap to avoid stack overflow
    auto oldW = std::make_unique<OldWeight>();
    auto newW = std::make_unique<NewWeight>();

    // Read old weight
    // Note: The original loader does some complex reading for codebook compression.
    // However, if we assume the input is the "uncompressed" binary dump or we just read it as raw struct?
    // Wait, the original code `Mix9svqWeightLoader::load` reads `feature_dwconv_weight` etc directly, 
    // BUT `read_compressed_mapping` reads codebook specially.
    // IF the user provides a raw dump of the memory (which is what `Weight` struct represents), then we can just read it.
    // BUT `mix9svqnnue.cpp` loads it from a file that might be compressed.
    // The `Weight` struct IS the in-memory representation.
    // If the file on disk matches the `Weight` struct layout (except for the compressed parts), we need to be careful.
    
    // Looking at `Mix9svqWeightLoader::load`:
    // 1. `read_compressed_mapping` -> reads codebook and mapping_index.
    // 2. `in.read` feature_dwconv_weight
    // 3. `in.read` feature_dwconv_bias
    // 4. `in.read` buckets
    
    // The `read_compressed_mapping` reads from the stream and fills `w.codebook`.
    // It seems the file format is:
    // [Compressed Codebook]
    // [Mapping Index]
    // [Feature DWConv Weight]
    // [Feature DWConv Bias]
    // [Buckets]
    
    // My converter needs to preserve the compressed codebook part?
    // Or does it read the fully loaded weight and write it back?
    // If I want to convert the FILE on disk, I need to handle the file format.
    // The file format starts with compressed codebook.
    // I should probably just copy the bytes of the compressed codebook part directly if I can determine its size.
    // But `read_compressed_mapping` logic is complex and bit-stream based.
    
    // Alternative: The user might want to convert the *loaded* weight? No, they want to convert the file so they can load it with the new engine.
    
    // The "Compressed Codebook" part does NOT depend on the `HeadBucket` layout.
    // So I can just read the file until I hit the `HeadBucket` part?
    // No, `HeadBucket` is at the end.
    // `feature_dwconv_weight` and `bias` are before `buckets`.
    
    // Strategy:
    // 1. Read the entire file into a buffer.
    // 2. Identify where `buckets` start.
    //    The file structure is:
    //    [Compressed Codebook Data] -> Variable size?
    //    [Mapping Index] -> Fixed size: sizeof(uint16_t) * 2 * ShapeNum
    //    [Feature DWConv Weight] -> Fixed size
    //    [Feature DWConv Bias] -> Fixed size
    //    [Buckets] -> Fixed size (Old Layout)
    
    // Wait, `read_compressed_mapping` reads `w.mapping_index` at the end of it.
    // `in.read(reinterpret_cast<char *>(&w.mapping_index[0][0]), sizeof(w.mapping_index));`
    
    // So the file format is:
    // [Bitstream for Codebook]
    // [Mapping Index (2 * 442503 * 2 bytes)]
    // [Feature DWConv Weight]
    // [Feature DWConv Bias]
    // [Buckets (Old)]
    
    // I don't know the size of the bitstream.
    // However, I know the size of everything AFTER the bitstream.
    // And I know the total file size.
    // So Size(Bitstream) = FileSize - Size(MappingIndex) - Size(DWConvW) - Size(DWConvB) - Size(BucketsOld).
    
    // So I can:
    // 1. Get file size.
    // 2. Calculate size of the "Tail" (MappingIndex + DWConv + BucketsOld).
    // 3. Read (FileSize - TailSize) bytes -> This is the header/codebook part. Write it to output as is.
    // 4. Read MappingIndex, DWConvW, DWConvB. Write them to output as is.
    // 5. Read BucketsOld. Convert to BucketsNew. Write to output.
    
    in.seekg(0, std::ios::end);
    size_t fileSize = in.tellg();
    in.seekg(0, std::ios::beg);
    
    size_t mappingIndexSize = sizeof(uint16_t) * 2 * ShapeNum;
    size_t dwConvWSize = sizeof(int16_t) * 9 * FeatDWConvDim;
    size_t dwConvBSize = sizeof(int16_t) * FeatDWConvDim;
    size_t bucketsOldSize = sizeof(OldWeight::HeadBucket) * NumHeadBucket;
    
    size_t tailSize = mappingIndexSize + dwConvWSize + dwConvBSize + bucketsOldSize;
    
    if (fileSize < tailSize) {
        std::cerr << "File too small to contain weight data." << std::endl;
        return 1;
    }
    
    size_t headerSize = fileSize - tailSize;
    
    // Copy header
    std::vector<char> header(headerSize);
    in.read(header.data(), headerSize);
    
    std::ofstream out(argv[2], std::ios::binary);
    out.write(header.data(), headerSize);
    
    // Copy Mapping Index, DWConvW, DWConvB
    size_t midSize = mappingIndexSize + dwConvWSize + dwConvBSize;
    std::vector<char> mid(midSize);
    in.read(mid.data(), midSize);
    out.write(mid.data(), midSize);
    
    // Convert Buckets
    for (int i = 0; i < NumHeadBucket; ++i) {
        OldWeight::HeadBucket oldB;
        in.read(reinterpret_cast<char*>(&oldB), sizeof(oldB));
        
        NewWeight::HeadBucket newB;
        // Copy all members up to policy_output_weight
        // We can use memcpy for the prefix
        // Offset of policy_output_weight in Old:
        size_t prefixSize = offsetof(OldWeight::HeadBucket, policy_output_weight);
        std::memcpy(&newB, &oldB, prefixSize);
        
        // Copy policy_output_weight
        std::memcpy(newB.policy_output_weight, oldB.policy_output_weight, sizeof(newB.policy_output_weight));
        
        // Copy policy_output_bias
        newB.policy_output_bias = oldB.policy_output_bias;
        
        // Zero padding
        std::memset(newB.__padding_to_64bytes_1, 0, sizeof(newB.__padding_to_64bytes_1));
        
        out.write(reinterpret_cast<char*>(&newB), sizeof(newB));
    }
    
    std::cout << "Conversion complete." << std::endl;
    
    return 0;
}
