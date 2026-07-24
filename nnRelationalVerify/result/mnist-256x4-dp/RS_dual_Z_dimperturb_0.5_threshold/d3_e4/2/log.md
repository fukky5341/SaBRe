## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0181036


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0044134, -0.0010648, -0.0044134, -0.0010648, -0.0033486, 0.0033486)
1: (-0.0028991, 0.0031835, -0.0028991, 0.0031835, -0.0060673, 0.0060673)
2: (0.0078537, 0.0214431, 0.0078537, 0.0214431, -0.0135894, 0.0135894)
3: (-0.0027016, 0.0040248, -0.0027016, 0.0040248, -0.0067264, 0.0067264)
4: (0.9885097, 1.0123649, 0.9885097, 1.0123649, -0.0238551, 0.0238551)
5: (-0.0041727, 0.0058704, -0.0041727, 0.0058704, -0.0100430, 0.0100430)
6: (-0.0117902, -0.0061656, -0.0117902, -0.0061656, -0.0056246, 0.0056246)
7: (-0.0103073, -0.0017661, -0.0103073, -0.0017661, -0.0085412, 0.0085412)
8: (-0.0065942, -0.0027080, -0.0065942, -0.0027080, -0.0038861, 0.0038861)
9: (-0.0046138, 0.0191426, -0.0046138, 0.0191426, -0.0200050, 0.0200050)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.94 + 2.65 = 4.59 seconds
status: Status.VERIFIED
relational distance
Output dim: 4, lower bound: -0.0152787, upper bound: 0.0152787
