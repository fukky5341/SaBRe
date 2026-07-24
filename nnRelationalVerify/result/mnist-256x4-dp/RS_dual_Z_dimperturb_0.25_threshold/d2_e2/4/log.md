## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0004263


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9877854, 0.9891267, 0.9877854, 0.9891267, -0.0011653, 0.0011653)
1: (-0.0043075, -0.0039733, -0.0043075, -0.0039733, -0.0002904, 0.0002904)
2: (0.0110023, 0.0127735, 0.0110023, 0.0127735, -0.0015388, 0.0015388)
3: (-0.0070871, -0.0062809, -0.0070871, -0.0062809, -0.0007004, 0.0007004)
4: (0.0026574, 0.0030002, 0.0026574, 0.0030002, -0.0002978, 0.0002978)
5: (0.0127974, 0.0150252, 0.0127974, 0.0150252, -0.0019354, 0.0019354)
6: (-0.0022727, -0.0017073, -0.0022727, -0.0017073, -0.0004912, 0.0004912)
7: (-0.0090179, -0.0075549, -0.0090179, -0.0075549, -0.0012709, 0.0012709)
8: (-0.0043066, -0.0035372, -0.0043066, -0.0035372, -0.0006684, 0.0006684)
9: (0.0022377, 0.0031298, 0.0022377, 0.0031298, -0.0007750, 0.0007750)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.26 = 2.53 seconds
status: Status.ADV_EXAMPLE
