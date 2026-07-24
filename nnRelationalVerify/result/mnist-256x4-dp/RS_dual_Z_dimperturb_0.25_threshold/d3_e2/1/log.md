## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00147475


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0005469, 0.0028201, 0.0005469, 0.0028201, -0.0013500, 0.0013500)
1: (0.0014013, 0.0017297, 0.0014013, 0.0017297, -0.0001950, 0.0001950)
2: (0.0128007, 0.0140575, 0.0128007, 0.0140575, -0.0007464, 0.0007464)
3: (-0.0014414, -0.0001415, -0.0014414, -0.0001415, -0.0007719, 0.0007719)
4: (-0.0038837, -0.0024766, -0.0038837, -0.0024766, -0.0008357, 0.0008357)
5: (0.0064571, 0.0077887, 0.0064571, 0.0077887, -0.0007908, 0.0007908)
6: (0.0033194, 0.0086029, 0.0033194, 0.0086029, -0.0031378, 0.0031378)
7: (-0.0142731, -0.0070774, -0.0142731, -0.0070774, -0.0042734, 0.0042734)
8: (0.9791596, 0.9842284, 0.9791596, 0.9842284, -0.0030103, 0.0030103)
9: (-0.0015709, 0.0030302, -0.0015709, 0.0030302, -0.0027325, 0.0027325)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.62 + 1.28 = 2.90 seconds
status: Status.ADV_EXAMPLE
