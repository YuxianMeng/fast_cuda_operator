/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

#include <torch/extension.h>
#include <vector>

/*
CPP Binding for CUDA OP
*/

// CUDA forward/backward declarations
torch::Tensor cuda_forward(
    torch::Tensor x,
    torch::Tensor d
);

std::vector<torch::Tensor> cuda_backward(
    const torch::Tensor grad_y,
    const torch::Tensor y,
    const torch::Tensor x,
    const torch::Tensor d,
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Input check and call to CUDA OP
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor d
) {
  CHECK_INPUT(x);
  CHECK_INPUT(d);
  return cuda_forward(x, d);
}

// todo add CHECK
std::vector<torch::Tensor> backward(
    const torch::Tensor grad_y,
    const torch::Tensor y,
    const torch::Tensor x,
    const torch::Tensor d,
) {
    CHECK_INPUT(grad_y);
    CHECK_INPUT(y);
    CHECK_INPUT(x);
    CHECK_INPUT(d);
    return cuda_backward(grad_y, y, x, d)
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward,
        "example cum_mul_sum forward (CUDA)");
  m.def("backward", &backward,
        "example cum_mul_sum forward (CUDA)");
}
