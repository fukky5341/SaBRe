## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.752e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041965, -0.0041909, -0.0041965, -0.0041909, -0.0000027, 0.0000027)
1: (-0.0098137, -0.0096063, -0.0098137, -0.0096063, -0.0001006, 0.0001006)
2: (0.9646866, 0.9649354, 0.9646866, 0.9649354, -0.0001208, 0.0001208)
3: (-0.0141601, -0.0123243, -0.0141601, -0.0123243, -0.0008908, 0.0008908)
4: (0.0002443, 0.0003839, 0.0002443, 0.0003839, -0.0000677, 0.0000677)
5: (0.0175173, 0.0176584, 0.0175173, 0.0176584, -0.0000685, 0.0000685)
6: (0.0033178, 0.0033864, 0.0033178, 0.0033864, -0.0000333, 0.0000333)
7: (-0.0045843, -0.0041085, -0.0045843, -0.0041085, -0.0002309, 0.0002309)
8: (0.0130922, 0.0134696, 0.0130922, 0.0134696, -0.0001831, 0.0001831)
9: (0.0212722, 0.0219510, 0.0212722, 0.0219510, -0.0003294, 0.0003294)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.12 = 2.38 seconds
status: Status.ADV_EXAMPLE
