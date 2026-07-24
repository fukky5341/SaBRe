## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00146637


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040755, -0.0037667, -0.0040755, -0.0037667, -0.0003088, 0.0003088)
1: (-0.0052817, -0.0043531, -0.0052817, -0.0043531, -0.0009286, 0.0009286)
2: (0.9696870, 0.9710149, 0.9696870, 0.9710149, -0.0013279, 0.0013279)
3: (0.0259543, 0.0325165, 0.0259543, 0.0325165, -0.0049469, 0.0049469)
4: (-0.0031661, -0.0025706, -0.0031661, -0.0025706, -0.0005955, 0.0005955)
5: (0.0138378, 0.0145749, 0.0138378, 0.0145749, -0.0007371, 0.0007371)
6: (0.0045494, 0.0050629, 0.0045494, 0.0050629, -0.0005135, 0.0005135)
7: (-0.0162052, -0.0145045, -0.0162052, -0.0145045, -0.0017007, 0.0017007)
8: (0.0038727, 0.0052219, 0.0038727, 0.0052219, -0.0013492, 0.0013492)
9: (0.0044849, 0.0071168, 0.0044849, 0.0071168, -0.0026319, 0.0026319)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 1.46 = 2.67 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0011950, upper bound: 0.0011950
