import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ternary_zero import Tensor, zeros, ones, randn, tensor, no_grad, is_grad_enabled
import ternary_zero as tz
from ternary_zero._backend import has_torch, to_numpy

if has_torch():
    import torch


def _to_numpy(data):
    return to_numpy(data)


class TestTensorCreation:
    def test_from_list(self):
        t = tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.dtype == np.float32

    def test_zeros(self):
        t = zeros(3, 4)
        assert t.shape == (3, 4)
        assert np.all(_to_numpy(t.data) == 0)

    def test_ones(self):
        t = ones(2, 3)
        assert t.shape == (2, 3)
        assert np.all(_to_numpy(t.data) == 1)

    def test_randn(self):
        t = randn(5, 5)
        assert t.shape == (5, 5)
        assert t.dtype == np.float32

    def test_requires_grad(self):
        t = tensor([1.0, 2.0], requires_grad=True)
        assert t.requires_grad
        assert t.is_leaf

    def test_detach(self):
        t = tensor([1.0, 2.0], requires_grad=True)
        d = t.detach()
        assert not d.requires_grad

    def test_clone(self):
        t = tensor([1.0, 2.0])
        c = t.clone()
        c_data = _to_numpy(c.data)
        c_data[0] = 99.0
        assert _to_numpy(t.data)[0] == 1.0


class TestTensorOps:
    def test_add(self):
        a = tensor([1.0, 2.0])
        b = tensor([3.0, 4.0])
        c = a + b
        assert np.allclose(_to_numpy(c.data), [4.0, 6.0])

    def test_sub(self):
        a = tensor([5.0, 6.0])
        b = tensor([1.0, 2.0])
        c = a - b
        assert np.allclose(_to_numpy(c.data), [4.0, 4.0])

    def test_mul(self):
        a = tensor([2.0, 3.0])
        b = tensor([4.0, 5.0])
        c = a * b
        assert np.allclose(_to_numpy(c.data), [8.0, 15.0])

    def test_div(self):
        a = tensor([10.0, 20.0])
        b = tensor([2.0, 4.0])
        c = a / b
        assert np.allclose(_to_numpy(c.data), [5.0, 5.0])

    def test_scalar_ops(self):
        a = tensor([1.0, 2.0])
        c = a + 10
        assert np.allclose(_to_numpy(c.data), [11.0, 12.0])
        d = a * 3
        assert np.allclose(_to_numpy(d.data), [3.0, 6.0])

    def test_matmul(self):
        a = tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        expected = np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[5.0, 6.0], [7.0, 8.0]])
        assert np.allclose(_to_numpy(c.data), expected)

    def test_sum(self):
        a = tensor([[1.0, 2.0], [3.0, 4.0]])
        s = a.sum()
        assert np.isclose(_to_numpy(s.data), 10.0)

    def test_mean(self):
        a = tensor([[1.0, 2.0], [3.0, 4.0]])
        m = a.mean()
        assert np.isclose(_to_numpy(m.data), 2.5)

    def test_relu(self):
        a = tensor([-1.0, 2.0, -3.0, 4.0])
        r = a.relu()
        assert np.allclose(_to_numpy(r.data), [0.0, 2.0, 0.0, 4.0])

    def test_softmax(self):
        a = tensor([1.0, 2.0, 3.0])
        s = a.softmax()
        assert np.isclose(_to_numpy(s.data).sum(), 1.0)
        assert np.all(_to_numpy(s.data) > 0)

    def test_view(self):
        a = tensor([[1.0, 2.0, 3.0, 4.0]])
        b = a.view(2, 2)
        assert b.shape == (2, 2)

    def test_transpose(self):
        a = tensor([[1.0, 2.0], [3.0, 4.0]])
        t = a.transpose(0, 1)
        assert t.shape == (2, 2)
        assert np.isclose(_to_numpy(t.data)[0, 1], 3.0)

    def test_neg(self):
        a = tensor([1.0, -2.0])
        c = -a
        assert np.allclose(_to_numpy(c.data), [-1.0, 2.0])


class TestAutograd:
    def test_simple_backward(self):
        x = tensor([2.0, 3.0], requires_grad=True)
        y = x.sum()
        y.backward()
        assert x.grad is not None
        assert np.allclose(_to_numpy(x.grad.data), [1.0, 1.0])

    def test_matmul_backward(self):
        x = tensor([[1.0, 2.0]], requires_grad=True)
        w = tensor([[3.0], [4.0]], requires_grad=True)
        y = x @ w
        y.backward()
        assert x.grad is not None
        assert w.grad is not None
        assert np.allclose(_to_numpy(x.grad.data), [[3.0, 4.0]])
        assert np.allclose(_to_numpy(w.grad.data), [[1.0], [2.0]])

    def test_mul_backward(self):
        x = tensor([2.0, 3.0], requires_grad=True)
        y = tensor([4.0, 5.0], requires_grad=True)
        z = x * y
        z = z.sum()
        z.backward()
        assert np.allclose(_to_numpy(x.grad.data), [4.0, 5.0])
        assert np.allclose(_to_numpy(y.grad.data), [2.0, 3.0])

    def test_relu_backward(self):
        x = tensor([-1.0, 2.0, -3.0], requires_grad=True)
        y = x.relu()
        y = y.sum()
        y.backward()
        assert np.allclose(_to_numpy(x.grad.data), [0.0, 1.0, 0.0])

    def test_no_grad(self):
        with no_grad():
            x = tensor([1.0, 2.0], requires_grad=True)
            assert not x.requires_grad

    def test_chained_ops(self):
        x = tensor([2.0], requires_grad=True)
        y = (x * x + x * 3 + 5)
        y.backward()
        assert np.isclose(_to_numpy(x.grad.data)[0], 7.0)


class TestNNModules:
    def test_linear(self):
        layer = tz.nn.Linear(3, 2)
        x = randn(1, 3)
        y = layer(x)
        assert y.shape == (1, 2)

    def test_linear_params(self):
        layer = tz.nn.Linear(3, 2)
        params = list(layer.parameters())
        assert len(params) == 2

    def test_sequential(self):
        model = tz.nn.Sequential(
            tz.nn.Linear(3, 4),
            tz.nn.ReLU(),
            tz.nn.Linear(4, 2),
        )
        x = randn(1, 3)
        y = model(x)
        assert y.shape == (1, 2)

    def test_bitlinear(self):
        layer = tz.nn.BitLinear(16, 8)
        x = randn(1, 16)
        y = layer(x)
        assert y.shape == (1, 8)

    def test_relu_module(self):
        relu = tz.nn.ReLU()
        x = tensor([-1.0, 2.0])
        y = relu(x)
        assert np.allclose(_to_numpy(y.data), [0.0, 2.0])

    def test_layernorm(self):
        ln = tz.nn.LayerNorm(4)
        x = randn(2, 4)
        y = ln(x)
        assert y.shape == (2, 4)

    def test_dropout_eval(self):
        dp = tz.nn.Dropout(p=0.5)
        dp.eval()
        x = tensor([1.0, 2.0, 3.0, 4.0])
        y = dp(x)
        assert np.allclose(_to_numpy(y.data), _to_numpy(x.data))

    def test_module_repr(self):
        model = tz.nn.Sequential(
            tz.nn.Linear(3, 4),
            tz.nn.ReLU(),
        )
        r = repr(model)
        assert "Sequential" in r

    def test_state_dict(self):
        layer = tz.nn.Linear(3, 2)
        state = layer.state_dict()
        assert "weight" in state
        assert "bias" in state

    def test_cross_entropy(self):
        loss_fn = tz.nn.CrossEntropyLoss()
        logits = tensor([[2.0, 1.0, 0.1]])
        target = tensor([0]).long()
        loss = loss_fn(logits, target)
        assert loss.data.ndim == 0 or (hasattr(loss.data, 'dim') and loss.data.dim() == 0)

    def test_mse_loss(self):
        loss_fn = tz.nn.MSELoss()
        pred = tensor([1.0, 2.0])
        target = tensor([1.5, 2.5])
        loss = loss_fn(pred, target)
        assert loss.data.ndim == 0 or (hasattr(loss.data, 'dim') and loss.data.dim() == 0)


class TestOptim:
    def test_sgd(self):
        x = tensor([1.0, 2.0], requires_grad=True)
        opt = tz.optim.SGD([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert np.allclose(_to_numpy(x.data), [0.8, 1.6])

    def test_adam(self):
        x = tensor([1.0, 2.0], requires_grad=True)
        opt = tz.optim.Adam([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(_to_numpy(x.data), [1.0, 2.0])

    def test_sgd_momentum(self):
        x = tensor([1.0], requires_grad=True)
        opt = tz.optim.SGD([x], lr=0.01, momentum=0.9)
        for _ in range(10):
            opt.zero_grad()
            y = (x * x).sum()
            y.backward()
            opt.step()
        assert _to_numpy(x.data)[0] < 1.0


class TestQuantize:
    def test_ternary_quantize(self):
        w = tensor([0.8, -0.7, 0.1, -0.05, 0.6, -0.9])
        ternary, scale = tz.quantize.ternary_quantize(w, alpha=0.5)
        assert scale > 0
        ternary_np = _to_numpy(ternary.data)
        assert set(np.unique(ternary_np)).issubset({-1, 0, 1})

    def test_pack_unpack_roundtrip(self):
        weights = tensor([1, 0, -1, 1, 0, 0, -1, 1, 0, 1, -1, 0, 1, -1, 0, 1])
        packed = tz.quantize.pack_ternary_to_u32(weights, 16)
        unpacked = tz.quantize.unpack_u32_to_ternary(packed, 16)
        assert np.array_equal(_to_numpy(weights.data).astype(np.int8), _to_numpy(unpacked.data))

    def test_ternary_quantize_fixed(self):
        w = tensor([0.5, -0.5, 0.1, -0.1, 0.0])
        t = tz.quantize.ternary_quantize_fixed(w, threshold=0.2)
        assert np.allclose(_to_numpy(t.data), [1, -1, 0, 0, 0])

    def test_dequantize(self):
        t = tensor([1, 0, -1, 1])
        d = tz.quantize.dequantize_ternary(t, 0.5)
        assert np.allclose(_to_numpy(d.data), [0.5, 0.0, -0.5, 0.5])


class TestConv:
    def test_conv2d_forward(self):
        layer = tz.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        x = randn(1, 3, 4, 4)
        y = layer(x)
        assert y.shape == (1, 8, 4, 4)

    def test_conv2d_no_padding(self):
        layer = tz.nn.Conv2d(1, 1, kernel_size=3)
        x = randn(1, 1, 5, 5)
        y = layer(x)
        assert y.shape == (1, 1, 3, 3)

    def test_conv2d_stride(self):
        layer = tz.nn.Conv2d(1, 1, kernel_size=3, stride=2)
        x = randn(1, 1, 6, 6)
        y = layer(x)
        assert y.shape == (1, 1, 2, 2)

    def test_conv2d_params(self):
        layer = tz.nn.Conv2d(3, 16, 3)
        params = list(layer.parameters())
        assert len(params) == 2

    def test_conv1d_forward(self):
        layer = tz.nn.Conv1d(3, 8, kernel_size=3, padding=1)
        x = randn(1, 3, 10)
        y = layer(x)
        assert y.shape == (1, 8, 10)

    def test_conv1d_no_padding(self):
        layer = tz.nn.Conv1d(1, 1, kernel_size=3)
        x = randn(1, 1, 8)
        y = layer(x)
        assert y.shape == (1, 1, 6)


class TestPooling:
    def test_maxpool2d(self):
        layer = tz.nn.MaxPool2d(kernel_size=2)
        x = tensor([[[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]]]])
        y = layer(x)
        assert y.shape == (1, 1, 2, 2)
        assert np.isclose(_to_numpy(y.data)[0, 0, 0, 0], 6.0)
        assert np.isclose(_to_numpy(y.data)[0, 0, 1, 1], 16.0)

    def test_maxpool2d_stride(self):
        layer = tz.nn.MaxPool2d(kernel_size=2, stride=1)
        x = randn(1, 1, 4, 4)
        y = layer(x)
        assert y.shape == (1, 1, 3, 3)

    def test_avgpool2d(self):
        layer = tz.nn.AvgPool2d(kernel_size=2)
        x = tensor([[[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]]]])
        y = layer(x)
        assert y.shape == (1, 1, 2, 2)
        assert np.isclose(_to_numpy(y.data)[0, 0, 0, 0], 3.5)

    def test_adaptive_avg_pool(self):
        layer = tz.nn.AdaptiveAvgPool2d(1)
        x = randn(1, 3, 8, 8)
        y = layer(x)
        assert y.shape == (1, 3, 1, 1)

    def test_adaptive_avg_pool_2x2(self):
        layer = tz.nn.AdaptiveAvgPool2d(2)
        x = randn(1, 3, 8, 8)
        y = layer(x)
        assert y.shape == (1, 3, 2, 2)

    def test_global_avg_pool(self):
        layer = tz.nn.GlobalAvgPool2d()
        x = randn(1, 3, 4, 4)
        y = layer(x)
        assert y.shape == (1, 3)


class TestEmbedding:
    def test_embedding_forward(self):
        emb = tz.nn.Embedding(10, 4)
        idx = tensor([0, 1, 2, 3])
        y = emb(idx)
        assert y.shape == (4, 4)

    def test_embedding_params(self):
        emb = tz.nn.Embedding(10, 4)
        params = list(emb.parameters())
        assert len(params) == 1
        assert params[0].shape == (10, 4)

    def test_embedding_padding(self):
        emb = tz.nn.Embedding(5, 3, padding_idx=0)
        idx = tensor([0])
        y = emb(idx)
        assert np.allclose(_to_numpy(y.data), 0.0)


class TestContainers:
    def test_module_list(self):
        layers = tz.nn.ModuleList([
            tz.nn.Linear(3, 4),
            tz.nn.ReLU(),
            tz.nn.Linear(4, 2),
        ])
        assert len(layers) == 3

    def test_module_list_append(self):
        layers = tz.nn.ModuleList()
        layers.append(tz.nn.Linear(3, 4))
        layers.append(tz.nn.ReLU())
        assert len(layers) == 2

    def test_module_list_iter(self):
        layers = tz.nn.ModuleList([
            tz.nn.Linear(3, 4),
            tz.nn.ReLU(),
        ])
        count = 0
        for _ in layers:
            count += 1
        assert count == 2

    def test_module_list_params(self):
        layers = tz.nn.ModuleList([
            tz.nn.Linear(3, 4),
            tz.nn.Linear(4, 2),
        ])
        params = list(layers.parameters())
        assert len(params) == 4

    def test_module_dict(self):
        modules = tz.nn.ModuleDict({
            "fc1": tz.nn.Linear(3, 4),
            "fc2": tz.nn.Linear(4, 2),
        })
        assert len(modules) == 2
        assert "fc1" in modules

    def test_module_dict_setitem(self):
        modules = tz.nn.ModuleDict()
        modules["fc"] = tz.nn.Linear(3, 4)
        assert len(modules) == 1

    def test_flatten(self):
        layer = tz.nn.Flatten(1)
        x = randn(2, 3, 4)
        y = layer(x)
        assert y.shape == (2, 12)

    def test_flatten_keep_last(self):
        layer = tz.nn.Flatten(1, 2)
        x = randn(2, 3, 4, 5)
        y = layer(x)
        assert y.shape == (2, 12, 5)


class TestLRScheduler:
    def test_step_lr(self):
        x = tensor([1.0], requires_grad=True)
        opt = tz.optim.SGD([x], lr=0.1)
        scheduler = tz.optim.StepLR(opt, step_size=2, gamma=0.5)
        assert np.isclose(opt.param_groups[0]["lr"], 0.1)
        scheduler.step()
        assert np.isclose(opt.param_groups[0]["lr"], 0.1)
        scheduler.step()
        assert np.isclose(opt.param_groups[0]["lr"], 0.05)

    def test_exponential_lr(self):
        x = tensor([1.0], requires_grad=True)
        opt = tz.optim.SGD([x], lr=1.0)
        scheduler = tz.optim.ExponentialLR(opt, gamma=0.9)
        assert np.isclose(opt.param_groups[0]["lr"], 1.0)
        scheduler.step()
        assert np.isclose(opt.param_groups[0]["lr"], 0.9)

    def test_cosine_annealing_lr(self):
        x = tensor([1.0], requires_grad=True)
        opt = tz.optim.Adam([x], lr=0.1)
        scheduler = tz.optim.CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        initial_lr = opt.param_groups[0]["lr"]
        assert initial_lr > 0
        for _ in range(10):
            scheduler.step()
        assert np.isclose(opt.param_groups[0]["lr"], 0.0, atol=1e-6)

    def test_linear_lr(self):
        x = tensor([1.0], requires_grad=True)
        opt = tz.optim.SGD([x], lr=1.0)
        scheduler = tz.optim.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=10)
        assert np.isclose(opt.param_groups[0]["lr"], 0.1)
        for _ in range(10):
            scheduler.step()
        assert np.isclose(opt.param_groups[0]["lr"], 1.0)

    def test_reduce_lr_on_plateau(self):
        x = tensor([1.0], requires_grad=True)
        opt = tz.optim.SGD([x], lr=1.0)
        scheduler = tz.optim.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2)
        scheduler.step(1.0)
        assert np.isclose(opt.param_groups[0]["lr"], 1.0)
        scheduler.step(1.0)
        assert np.isclose(opt.param_groups[0]["lr"], 1.0)
        scheduler.step(1.0)
        assert np.isclose(opt.param_groups[0]["lr"], 0.1)


class TestAdagrad:
    def test_adagrad_step(self):
        x = tensor([1.0, 2.0], requires_grad=True)
        opt = tz.optim.Adagrad([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(_to_numpy(x.data), [1.0, 2.0])


class TestNewOptimizers:
    def test_adamw(self):
        x = tensor([1.0, 2.0], requires_grad=True)
        opt = tz.optim.AdamW([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(_to_numpy(x.data), [1.0, 2.0])

    def test_rmsprop(self):
        x = tensor([1.0, 2.0], requires_grad=True)
        opt = tz.optim.RMSprop([x], lr=0.01)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(_to_numpy(x.data), [1.0, 2.0])


class TestGPUAcceleration:
    def test_backend_info(self):
        assert isinstance(has_torch(), bool)
        assert isinstance(tz.is_cuda_available(), bool)

    def test_device_property(self):
        t = tensor([1.0, 2.0])
        assert isinstance(t.device, str)

    def test_to_device(self):
        t = tensor([1.0, 2.0])
        t2 = t.to("cpu")
        assert t2.device == "cpu"
        assert np.allclose(_to_numpy(t2.data), [1.0, 2.0])

    def test_dtype_property(self):
        t = tensor([1.0, 2.0])
        assert t.dtype == np.float32

    def test_no_grad_context(self):
        with no_grad():
            t = tensor([1.0, 2.0], requires_grad=True)
            assert not t.requires_grad

    def test_enable_grad_context(self):
        from ternary_zero import enable_grad
        with no_grad():
            with enable_grad():
                t = tensor([1.0, 2.0], requires_grad=True)
                assert t.requires_grad

    @pytest.mark.skipif(not has_torch(), reason="PyTorch not available")
    def test_torch_backend_active(self):
        assert has_torch()
        t = tensor([1.0, 2.0])
        assert isinstance(t.data, torch.Tensor)

    @pytest.mark.skipif(not has_torch() or not tz.is_cuda_available(), reason="CUDA not available")
    def test_cuda_tensor(self):
        t = tensor([1.0, 2.0])
        t_cuda = t.cuda()
        assert "cuda" in t_cuda.device

    @pytest.mark.skipif(not has_torch() or not tz.is_cuda_available(), reason="CUDA not available")
    def test_cuda_linear(self):
        layer = tz.nn.Linear(3, 2)
        layer.cuda()
        x = randn(1, 3).cuda()
        y = layer(x)
        assert "cuda" in y.device


class TestDataPipeline:
    def test_tensor_dataset(self):
        x_data = randn(10, 3)
        y_data = randn(10, 2)
        dataset = tz.data.TensorDataset(x_data, y_data)
        assert len(dataset) == 10

    def test_dataloader(self):
        x_data = randn(10, 3)
        y_data = randn(10, 2)
        dataset = tz.data.TensorDataset(x_data, y_data)
        loader = tz.data.DataLoader(dataset, batch_size=2, shuffle=True)
        assert len(loader) == 5




class TestConvDtypeRegression:
    """Regression: conv weight params must stay float32 after He init."""

    def test_conv2d_weight_dtype_float32(self):
        layer = tz.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        w = _to_numpy(layer.weight.data)
        assert w.dtype == np.float32, "Conv2d weight dtype is {}, expected float32".format(w.dtype)

    def test_conv1d_weight_dtype_float32(self):
        layer = tz.nn.Conv1d(3, 8, kernel_size=3, padding=1)
        w = _to_numpy(layer.weight.data)
        assert w.dtype == np.float32, "Conv1d weight dtype is {}, expected float32".format(w.dtype)

    def test_conv2d_forward_dtype_match(self):
        layer = tz.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        x = randn(1, 3, 4, 4)
        y = layer(x)
        input_dtype = _to_numpy(x.data).dtype
        weight_dtype = _to_numpy(layer.weight.data).dtype
        output_dtype = _to_numpy(y.data).dtype
        assert weight_dtype == input_dtype, "Weight {} != input {}".format(weight_dtype, input_dtype)
        assert output_dtype == input_dtype, "Output {} != input {}".format(output_dtype, input_dtype)

    def test_conv1d_forward_dtype_match(self):
        layer = tz.nn.Conv1d(3, 8, kernel_size=3, padding=1)
        x = randn(1, 3, 10)
        y = layer(x)
        input_dtype = _to_numpy(x.data).dtype
        weight_dtype = _to_numpy(layer.weight.data).dtype
        output_dtype = _to_numpy(y.data).dtype
        assert weight_dtype == input_dtype, "Weight {} != input {}".format(weight_dtype, input_dtype)
        assert output_dtype == input_dtype, "Output {} != input {}".format(output_dtype, input_dtype)

    def test_conv2d_bias_dtype_float32(self):
        layer = tz.nn.Conv2d(3, 8, kernel_size=3)
        assert _to_numpy(layer.bias.data).dtype == np.float32

    def test_conv1d_bias_dtype_float32(self):
        layer = tz.nn.Conv1d(3, 8, kernel_size=3)
        assert _to_numpy(layer.bias.data).dtype == np.float32


class TestNativeExtension:
    """Tests for the Rust native extension (_core) import and dispatch."""

    def test_core_import_or_skip(self):
        try:
            from ternary_zero import _core
            assert callable(_core.has_cuda)
        except ImportError:
            pytest.skip("_core native extension not built (run: maturin develop --release)")

    def test_has_native_flag(self):
        assert isinstance(tz._HAS_NATIVE, bool)

    def test_fallback_works_without_native(self):
        w = tz.tensor([0.8, -0.7, 0.1])
        assert w.shape == (3,)
        assert w.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
