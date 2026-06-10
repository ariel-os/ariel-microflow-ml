# ariel-microflow-ml

Generic Tiny ML pipeline for [Ariel OS](https://ariel-os.github.io/ariel-os/)
Using [MicroFlow](https://github.com/matteocarnelos/microflow-rs)
as the inference engine for full Rust, `no_std` Ariel compatible, quantized TFLite models.

Ships with two example models: **LeNet5** (MNIST digits) trained from either
TensorFlow or PyTorch, and **MobileNetV1** (person detection) from the MicroFlow repo.

- MicroFlow paper: <https://arxiv.org/pdf/2409.19432>

## Prerequisites

- Rust toolchain (stable + the embedded target your board needs)
- [Ariel OS getting started](https://ariel-os.github.io/ariel-os/dev/docs/book/getting-started.html) (`laze`, `probe-rs`, etc.)
- [uv](https://docs.astral.sh/uv/) for the Python notebooks (env and dependences management)

## Workflow

### 1. Train and export a `.tflite`

TensorFlow and PyTorch are kept in separate uv projects to avoid version conflicts.
Pick one(here TensorFlow for example), then:

```
cd building_tf      # or building_torch
uv sync
```

Run the notebook. It downloads MNIST automatically (Keras /
torchvision cache) and writes an INT8-quantized model into `models/` at the
repo root — exactly where `src/main.rs`'s `#[model("models/…")]` attribute
looks for it.

> Only **quantized** models are supported. The notebook uses per-tensor INT8
> quantization, which is what MicroFlow expects.

`models_provided/mobilenetv1.tflite` is the pre-baked person-detection model
from MicroFlow's own repo — some ops (padding, batch-norm) are baked into the
weights so MicroFlow can consume it.

### 2. Build and flash

```
laze build -b <ariel-board-id> run --features <model-feature>
```

Available model features (pick exactly one):

| feature        | model                                                      |
|----------------|------------------------------------------------------------|
| `lenet5qtf`    | LeNet5 trained with TensorFlow (`models/lenet5_quantized_tf.tflite`)    |
| `lenet5qtorch` | LeNet5 trained with PyTorch (`models/lenet5_quantized_torch.tflite`)    |
| `mobilenetv1`  | Person detection (`models_provided/mobilenetv1.tflite`)                 |

Example, on a Raspberry Pi Pico 2 W:

```
laze build -b rpi-pico2-w run --features lenet5qtf
```

Stack sizes for the main thread are tuned in `src/main.rs` and
`laze-project.yml`.
 MobileNetV1 in particular needs 320 KiB.

## Benchmarking

Inspect the built binary:

```
arm-none-eabi-size build/bin/<board>/cargo/<target>/release/ariel-microflow-ml
nm --print-size --size-sort --demangle=rust --radix=d <same path>
```

Per-inference timing is logged at runtime (`info!` lines, averaged over 4 runs).

Reference numbers on a Raspberry Pi Pico 2 W, averaged over 4 runs ; thread stack is the minimum to set:

| model       | time (ms) | main thread stack (KiB) |
|-------------|-----------|-------------------------|
| lenet5q     | 53        | 14                      |
| mobilenetv1 | 3523      | 320                     |

## MicroFlow op support

MicroFlow only implements a subset of TFLite ops (Conv2D, DepthwiseConv2D,
FullyConnected, AveragePool2D, Reshape, Softmax, Transpose at the current day ). If your model uses anything else, the build will fail at the
`#[model(...)]` macro with an "unimplemented operator" message.

See the upstream repo for the current list and to contribute new ops:
<https://github.com/matteocarnelos/microflow-rs>

---

Notebook outputs are stripped at commit time via a `nbstripout` git filter — after cloning, run `pipx install nbstripout && nbstripout --install` once.
