#include "simdops.h"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>
#include <iomanip>

// Mock definitions from mix9svqnnue.h
constexpr int FeatureDim = 64;
constexpr int ValueDim   = 64;

// Use AVX2 alignment
constexpr int Alignment = 32;
constexpr simd::InstructionType IT = simd::AVX2;

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

template <int N, typename T>
using Batch = simd::detail::VecBatch<N, T, IT>;

// Original linear function (copied from mix9svqnnue.cpp or simdops usage)
// Since `simd::linear` is in `simdops.h`, we can just use it.
// Wait, `starBlock` uses `simd::linear`.
// I need to copy `starBlock` from `mix9svqnnue.cpp`.

// ----------------------------------------------------------------------------
// Original starBlock (copied from mix9svqnnue.cpp before my changes)
// ----------------------------------------------------------------------------
template <int OutSize, int InSize>
inline void
starBlock_original(int8_t output[OutSize], int8_t input[InSize], const StarBlockWeight<OutSize, InSize> &w)
{
    alignas(Alignment) int32_t upi32[OutSize * 2];
    alignas(Alignment) int8_t  up1[OutSize * 2], up2[OutSize * 2];

    // Corner Up 1
    simd::linear<OutSize * 2, InSize>(upi32, input, w.value_corner_up1.weight, w.value_corner_up1.bias);
    simd::crelu<OutSize * 2, 128>(up1, upi32);

    // Corner Up 2
    simd::linear<OutSize * 2, InSize>(upi32, input, w.value_corner_up2.weight, w.value_corner_up2.bias);
    simd::crelu<OutSize * 2, 128, true>(up2, upi32);

    alignas(Alignment) int8_t dotsum[OutSize];
    simd::dot2<OutSize, 128>(dotsum, up1, up2);

    alignas(Alignment) int32_t outputi32[OutSize];
    // Corner Down
    simd::linear<OutSize, OutSize, true>(outputi32, dotsum, w.value_corner_down.weight, w.value_corner_down.bias);
    simd::crelu<OutSize, 128>(output, outputi32);
}

// ----------------------------------------------------------------------------
// New linear4 and starBlock4 (copied from my changes in mix9svqnnue.cpp)
// ----------------------------------------------------------------------------

template <int OutSize, int InSize, bool SignedInput = false>
inline void linear4(int32_t *output0,
                    int32_t *output1,
                    int32_t *output2,
                    int32_t *output3,
                    const int8_t *input0,
                    const int8_t *input1,
                    const int8_t *input2,
                    const int8_t *input3,
                    const int8_t *weight,
                    const int32_t *bias)
{
    using I8LS  = simd::detail::VecLoadStore<int8_t, Alignment, IT>;
    using I32LS = simd::detail::VecLoadStore<int32_t, Alignment, IT>;
    using I8Op  = simd::detail::VecOp<int8_t, IT>;
    using I32Op = simd::detail::VecOp<int32_t, IT>;

    using OutB = Batch<OutSize, int32_t>;
    using InB  = Batch<InSize, int8_t>;

    typename I32Op::R acc0[OutB::NumBatch];
    typename I32Op::R acc1[OutB::NumBatch];
    typename I32Op::R acc2[OutB::NumBatch];
    typename I32Op::R acc3[OutB::NumBatch];

    for (int j = 0; j < OutB::NumBatch; j++) {
        auto b  = I32LS::load(bias + j * OutB::RegWidth);
        acc0[j] = b;
        acc1[j] = b;
        acc2[j] = b;
        acc3[j] = b;
    }

    constexpr int ChunkSize = 4;
    constexpr int NumChunks = InSize / ChunkSize;
    static_assert(InSize % ChunkSize == 0, "InSize must be a multiple of ChunkSize=4");

    const auto input0_32 = reinterpret_cast<const int32_t *>(input0);
    const auto input1_32 = reinterpret_cast<const int32_t *>(input1);
    const auto input2_32 = reinterpret_cast<const int32_t *>(input2);
    const auto input3_32 = reinterpret_cast<const int32_t *>(input3);

    for (int i = 0; i < NumChunks; i++) {
        auto in0 = typename I8Op::R(I32Op::set1(input0_32[i]));
        auto in1 = typename I8Op::R(I32Op::set1(input1_32[i]));
        auto in2 = typename I8Op::R(I32Op::set1(input2_32[i]));
        auto in3 = typename I8Op::R(I32Op::set1(input3_32[i]));

        auto wBase = reinterpret_cast<const typename I8Op::R *>(weight + i * OutSize * ChunkSize);

        for (int j = 0; j < OutB::NumBatch; j++) {
            auto w = I8LS::load(&wBase[j]);
            if constexpr (SignedInput) {
                I8Op::dot4_i8i8_accum(acc0[j], in0, w);
                I8Op::dot4_i8i8_accum(acc1[j], in1, w);
                I8Op::dot4_i8i8_accum(acc2[j], in2, w);
                I8Op::dot4_i8i8_accum(acc3[j], in3, w);
            } else {
                I8Op::dot4_u7i8_accum(acc0[j], in0, w);
                I8Op::dot4_u7i8_accum(acc1[j], in1, w);
                I8Op::dot4_u7i8_accum(acc2[j], in2, w);
                I8Op::dot4_u7i8_accum(acc3[j], in3, w);
            }
        }
    }

    for (int j = 0; j < OutB::NumBatch; j++) {
        I32LS::store(output0 + j * OutB::RegWidth, acc0[j]);
        I32LS::store(output1 + j * OutB::RegWidth, acc1[j]);
        I32LS::store(output2 + j * OutB::RegWidth, acc2[j]);
        I32LS::store(output3 + j * OutB::RegWidth, acc3[j]);
    }
}

template <int OutSize, int InSize>
inline void starBlock4(int8_t *output0,
                       int8_t *output1,
                       int8_t *output2,
                       int8_t *output3,
                       int8_t *input0,
                       int8_t *input1,
                       int8_t *input2,
                       int8_t *input3,
                       const StarBlockWeight<OutSize, InSize> &w)
{
    alignas(Alignment) int32_t upi32_0[OutSize * 2];
    alignas(Alignment) int32_t upi32_1[OutSize * 2];
    alignas(Alignment) int32_t upi32_2[OutSize * 2];
    alignas(Alignment) int32_t upi32_3[OutSize * 2];

    alignas(Alignment) int8_t up1_0[OutSize * 2], up2_0[OutSize * 2];
    alignas(Alignment) int8_t up1_1[OutSize * 2], up2_1[OutSize * 2];
    alignas(Alignment) int8_t up1_2[OutSize * 2], up2_2[OutSize * 2];
    alignas(Alignment) int8_t up1_3[OutSize * 2], up2_3[OutSize * 2];

    // Corner Up 1
    linear4<OutSize * 2, InSize, false>(upi32_0,
                                        upi32_1,
                                        upi32_2,
                                        upi32_3,
                                        input0,
                                        input1,
                                        input2,
                                        input3,
                                        w.value_corner_up1.weight,
                                        w.value_corner_up1.bias);

    simd::crelu<OutSize * 2, 128>(up1_0, upi32_0);
    simd::crelu<OutSize * 2, 128>(up1_1, upi32_1);
    simd::crelu<OutSize * 2, 128>(up1_2, upi32_2);
    simd::crelu<OutSize * 2, 128>(up1_3, upi32_3);

    // Corner Up 2
    linear4<OutSize * 2, InSize, false>(upi32_0,
                                        upi32_1,
                                        upi32_2,
                                        upi32_3,
                                        input0,
                                        input1,
                                        input2,
                                        input3,
                                        w.value_corner_up2.weight,
                                        w.value_corner_up2.bias);

    simd::crelu<OutSize * 2, 128, true>(up2_0, upi32_0);
    simd::crelu<OutSize * 2, 128, true>(up2_1, upi32_1);
    simd::crelu<OutSize * 2, 128, true>(up2_2, upi32_2);
    simd::crelu<OutSize * 2, 128, true>(up2_3, upi32_3);

    alignas(Alignment) int8_t dotsum0[OutSize];
    alignas(Alignment) int8_t dotsum1[OutSize];
    alignas(Alignment) int8_t dotsum2[OutSize];
    alignas(Alignment) int8_t dotsum3[OutSize];

    simd::dot2<OutSize, 128>(dotsum0, up1_0, up2_0);
    simd::dot2<OutSize, 128>(dotsum1, up1_1, up2_1);
    simd::dot2<OutSize, 128>(dotsum2, up1_2, up2_2);
    simd::dot2<OutSize, 128>(dotsum3, up1_3, up2_3);

    alignas(Alignment) int32_t outputi32_0[OutSize];
    alignas(Alignment) int32_t outputi32_1[OutSize];
    alignas(Alignment) int32_t outputi32_2[OutSize];
    alignas(Alignment) int32_t outputi32_3[OutSize];

    // Corner Down (SignedInput=true)
    linear4<OutSize, OutSize, true>(outputi32_0,
                                    outputi32_1,
                                    outputi32_2,
                                    outputi32_3,
                                    dotsum0,
                                    dotsum1,
                                    dotsum2,
                                    dotsum3,
                                    w.value_corner_down.weight,
                                    w.value_corner_down.bias);

    simd::crelu<OutSize, 128>(output0, outputi32_0);
    simd::crelu<OutSize, 128>(output1, outputi32_1);
    simd::crelu<OutSize, 128>(output2, outputi32_2);
    simd::crelu<OutSize, 128>(output3, outputi32_3);
}

// ----------------------------------------------------------------------------
// Test Harness
// ----------------------------------------------------------------------------

void fill_random_i8(int8_t* data, int size) {
    static std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (int i = 0; i < size; ++i) data[i] = (int8_t)dist(rng);
}

void fill_random_i32(int32_t* data, int size) {
    static std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (int i = 0; i < size; ++i) data[i] = dist(rng);
}

template <int OutSize, int InSize>
void fill_random_weights(StarBlockWeight<OutSize, InSize>& w) {
    fill_random_i8(w.value_corner_up1.weight, OutSize * 2 * InSize);
    fill_random_i32(w.value_corner_up1.bias, OutSize * 2);
    fill_random_i8(w.value_corner_up2.weight, OutSize * 2 * InSize);
    fill_random_i32(w.value_corner_up2.bias, OutSize * 2);
    fill_random_i8(w.value_corner_down.weight, OutSize * OutSize);
    fill_random_i32(w.value_corner_down.bias, OutSize);
}

int main() {
    std::cout << "Running Mix9SVQ Correctness Test..." << std::endl;

    // Allocate aligned memory
    alignas(Alignment) int8_t input0[FeatureDim];
    alignas(Alignment) int8_t input1[FeatureDim];
    alignas(Alignment) int8_t input2[FeatureDim];
    alignas(Alignment) int8_t input3[FeatureDim];

    alignas(Alignment) int8_t output0_ref[ValueDim];
    alignas(Alignment) int8_t output1_ref[ValueDim];
    alignas(Alignment) int8_t output2_ref[ValueDim];
    alignas(Alignment) int8_t output3_ref[ValueDim];

    alignas(Alignment) int8_t output0_new[ValueDim];
    alignas(Alignment) int8_t output1_new[ValueDim];
    alignas(Alignment) int8_t output2_new[ValueDim];
    alignas(Alignment) int8_t output3_new[ValueDim];

    StarBlockWeight<ValueDim, FeatureDim> weights;

    // Fill with random data
    fill_random_i8(input0, FeatureDim);
    fill_random_i8(input1, FeatureDim);
    fill_random_i8(input2, FeatureDim);
    fill_random_i8(input3, FeatureDim);
    fill_random_weights(weights);

    // Run Reference (Sequential)
    starBlock_original<ValueDim, FeatureDim>(output0_ref, input0, weights);
    starBlock_original<ValueDim, FeatureDim>(output1_ref, input1, weights);
    starBlock_original<ValueDim, FeatureDim>(output2_ref, input2, weights);
    starBlock_original<ValueDim, FeatureDim>(output3_ref, input3, weights);

    // Run New (Batched)
    starBlock4<ValueDim, FeatureDim>(output0_new, output1_new, output2_new, output3_new,
                                     input0, input1, input2, input3, weights);

    // Compare
    bool pass = true;
    for (int i = 0; i < ValueDim; ++i) {
        if (output0_ref[i] != output0_new[i]) {
            std::cout << "Mismatch in Output 0 at index " << i << ": Ref=" << (int)output0_ref[i] << " New=" << (int)output0_new[i] << std::endl;
            pass = false;
        }
        if (output1_ref[i] != output1_new[i]) {
            std::cout << "Mismatch in Output 1 at index " << i << ": Ref=" << (int)output1_ref[i] << " New=" << (int)output1_new[i] << std::endl;
            pass = false;
        }
        if (output2_ref[i] != output2_new[i]) {
            std::cout << "Mismatch in Output 2 at index " << i << ": Ref=" << (int)output2_ref[i] << " New=" << (int)output2_new[i] << std::endl;
            pass = false;
        }
        if (output3_ref[i] != output3_new[i]) {
            std::cout << "Mismatch in Output 3 at index " << i << ": Ref=" << (int)output3_ref[i] << " New=" << (int)output3_new[i] << std::endl;
            pass = false;
        }
    }

    if (pass) {
        std::cout << "Test PASSED: All outputs match exactly." << std::endl;
        return 0;
    } else {
        std::cout << "Test FAILED: Outputs do not match." << std::endl;
        return 1;
    }
}
