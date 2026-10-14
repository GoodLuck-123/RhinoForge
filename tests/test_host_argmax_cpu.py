"""Execute the production CPU helper without loading the backend or an device.

On a loaded ARM backend, reuse ``check_host_argmax_cpu(torch.ops.rpu.argmax_lastdim_host)``
to check the same cases through the real dispatcher and NEON implementation.
"""
from pathlib import Path
import shutil
import subprocess

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def host_argmax_cpu_cases():
    """Shared scalar, NEON boundary/tail, and real-vocabulary semantic cases."""
    for dtype in (torch.float16, torch.float32):
        for width in (1, 7, 8, 9, 13, 16, 17, 65, 248320):
            x = torch.randn((13, width), generator=torch.Generator().manual_seed(width), dtype=dtype)
            x[0].fill_(-float("inf"))
            x[1].fill_(float("inf"))
            x[2].zero_()
            x[2, ::2] = -0.0
            x[3].fill_(-0.0)
            x[3, ::2] = 0.0
            x[4].fill_(-1)
            x[4, sorted({0, min(3, width - 1), min(8, width - 1), width - 1})] = 5
            x[5].fill_(-1)
            x[5, sorted({min(3, width - 1), min(8, width - 1), width - 1})] = 5
            x[6, min(3, width - 1)] = float("nan")
            x[6, -1] = float("nan")
            x[7].fill_(float("inf"))
            x[7, -1] = -float("nan")
            x[8].fill_(float("nan"))
            x[9].fill_(-1)
            x[9, -1] = 5
            # Distinct FP32 values that become an FP16 tie must stay distinct.
            x[10].zero_()
            x[10, 0] = 1
            x[10, -1] = torch.nextafter(torch.tensor(1., dtype=dtype), torch.tensor(float("inf"), dtype=dtype))
            x[11].zero_()
            x[11, -1] = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype))
            yield f"{dtype}-width{width}", x
        yield f"{dtype}-zero-rows", torch.empty((0, 13), dtype=dtype)


def check_host_argmax_cpu(op):
    """Reusable actual-op check; no backend import, device creation, or timing."""
    checked = 0
    for name, x in host_argmax_cpu_cases():
        before = x.clone()
        expected = x.argmax(dim=-1)
        actual = op(x)
        assert actual.device.type == "cpu" and actual.dtype == torch.long, name
        assert actual.shape == expected.shape and torch.equal(actual, expected), name
        assert torch.equal(x.view(torch.uint8), before.view(torch.uint8)), name
        checked += 1
    invalid = (
        (torch.zeros((3, 4)).t(), "contiguous"),
        (torch.empty((1, 0)), "empty last dimension"),
        (torch.zeros(4), "2-D"),
        (torch.zeros((1, 2, 3)), "2-D"),
        (torch.tensor(1.), "2-D"),
        (torch.zeros((1, 4), dtype=torch.float64), "fp16 or fp32"),
        (torch.zeros((1, 4), dtype=torch.bfloat16), "fp16 or fp32"),
        (torch.zeros((1, 4), dtype=torch.long), "fp16 or fp32"),
    )
    for x, message in invalid:
        try:
            op(x)
        except RuntimeError as error:
            assert message in str(error), (message, str(error))
        else:
            raise AssertionError(f"argmax accepted invalid {x.dtype}/{tuple(x.shape)}/{x.stride()}")
    return {"valid_cases": checked, "rejected_cases": len(invalid)}


@pytest.fixture(scope="module")
def native_cpu_argmax(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    build = tmp_path_factory.mktemp("host-argmax-cpu")
    source = build / "argmax.cpp"
    binary = build / "argmax"
    source.write_text(r'''
#include "core/rpu_host_argmax.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    at::set_num_threads(2);
    const std::string dtype = argv[1];
    const int rank = std::stoi(argv[2]);
    const int64_t rows = std::stoll(argv[3]), cols = std::stoll(argv[4]);
    at::Tensor input = at::empty({rows, cols}, at::kFloat);
    std::cin.read(reinterpret_cast<char*>(input.data_ptr<float>()), input.nbytes());
    if (dtype == "half") input = input.to(at::kHalf);
    if (dtype == "double") input = input.to(at::kDouble);
    if (dtype == "bfloat16") input = input.to(at::kBFloat16);
    if (dtype == "long") input = input.to(at::kLong);
    if (rank == 0) input = input.reshape({});
    if (rank == 1) input = input.reshape({cols});
    if (rank == 3) input = input.reshape({1, rows, cols});
    if (std::string(argv[5]) == "strided") {
      auto strided = at::empty({rows, cols * 2}, input.options()).slice(1, 0, cols * 2, 2);
      strided.copy_(input);
      input = strided;
    }
    auto before = input.clone();
    auto output = rpu::host_argmax::argmax_lastdim_cpu(input);
    TORCH_CHECK(output.device().is_cpu() && output.scalar_type() == at::kLong &&
                output.dim() == 1 && output.size(0) == rows && output.is_contiguous(),
                "native host argmax returned wrong metadata");
    TORCH_CHECK(input.nbytes() == 0 ||
                std::memcmp(input.data_ptr(), before.data_ptr(), input.nbytes()) == 0,
                "native host argmax changed its input bytes");
    std::cout.write(reinterpret_cast<const char*>(output.data_ptr<int64_t>()), output.nbytes());
    return 0;
  } catch (const c10::Error& error) {
    std::cerr << error.what_without_backtrace();
    return 2;
  }
}
''')
    torch_root = Path(torch.__file__).resolve().parent
    command = [compiler, "-std=c++17", "-O3", "-pthread",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               "-I", str(ROOT / "src"), "-I", str(torch_root / "include"),
               str(source), "-L", str(torch_root / "lib"),
               f"-Wl,-rpath,{torch_root / 'lib'}", "-ltorch_cpu", "-lc10", "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr

    def run(x):
        cols = x.shape[-1] if x.ndim else 1
        rows = x.numel() // cols if cols else 1
        names = {torch.float16: "half", torch.float32: "float", torch.float64: "double",
                 torch.bfloat16: "bfloat16", torch.long: "long"}
        result = subprocess.run([str(binary), names[x.dtype], str(x.ndim), str(rows), str(cols),
                                 "contiguous" if x.is_contiguous() else "strided"],
                                input=x.float().contiguous().numpy().tobytes(),
                                capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.decode())
        return (torch.frombuffer(bytearray(result.stdout), dtype=torch.long).clone()
                if result.stdout else torch.empty(0, dtype=torch.long))
    return run


def test_cpu_native_argmax_matches_torch_and_rejects_invalid_inputs(native_cpu_argmax):
    check_host_argmax_cpu(native_cpu_argmax)
