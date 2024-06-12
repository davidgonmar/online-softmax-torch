#include <torch/extension.h>
#include <cub/block/block_reduce.cuh>


constexpr int THREADBLOCK_SIZE = 1024;

struct __align__(8) OnlineSoftmaxInfo { // todo - why does align 8 make it like 10% faster?
    float m;
    float d;
};

__device__ __forceinline__ OnlineSoftmaxInfo& reduce(OnlineSoftmaxInfo &a, OnlineSoftmaxInfo &b) {
    OnlineSoftmaxInfo res;
    float new_m = __max(a.m, b.m);
    res.m = new_m;
    res.d = a.d * __expf(a.m - new_m) + b.d * __expf(b.m - new_m);
    return res;
}

__global__ void online_softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const size_t num_features
) {
    // input of shape (batch_size, num_features)
    // output of shape (batch_size, num_features)

    const size_t batch_idx = blockIdx.x;
    
    typedef cub::BlockReduce<OnlineSoftmaxInfo, THREADBLOCK_SIZE> block_reduce;
    __shared__ OnlineSoftmaxInfo final_info;
    __shared__ typename block_reduce::TempStorage temp_storage;
    

    OnlineSoftmaxInfo info;
    info.m = FLT_MIN;
    info.d = 0.0;

    // reduce in each block
    for (size_t feature_idx = threadIdx.x; feature_idx < num_features; feature_idx += blockDim.x) {
        OnlineSoftmaxInfo new_info;
        new_info.m = input[batch_idx * num_features + feature_idx];
        new_info.d = 1.0;
        info = reduce(info, new_info);
    }

    info = block_reduce(temp_storage).Reduce(info, reduce);

    // broadcast the result to all threads
    if (threadIdx.x == 0) {
        final_info = info;
    }
    __syncthreads();


    // compute the output as out = exp(input - m) / d
    for (size_t feature_idx = threadIdx.x; feature_idx < num_features; feature_idx += blockDim.x) {
        const size_t input_idx = batch_idx * num_features + feature_idx;
        output[input_idx] = __expf(input[input_idx] - final_info.m) / final_info.d;
    }
}



void online_softmax_forward(
    const torch::Tensor& input,
    torch::Tensor& output
) { 
    const size_t num_features = input.size(1);

    const size_t num_blocks = input.size(0);

    online_softmax_forward_kernel<<<num_blocks, THREADBLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_features
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &online_softmax_forward, "Online softmax forward");
}