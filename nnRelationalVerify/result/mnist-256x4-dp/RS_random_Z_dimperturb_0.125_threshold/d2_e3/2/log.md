## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0002144


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9913451, 0.9921957, 0.9913451, 0.9921957, -0.0004605, 0.0004605)
1: (-0.0034205, -0.0032086, -0.0034205, -0.0032086, -0.0001148, 0.0001148)
2: (0.0069497, 0.0080730, 0.0069497, 0.0080730, -0.0006081, 0.0006081)
3: (-0.0049476, -0.0044363, -0.0049476, -0.0044363, -0.0002768, 0.0002768)
4: (0.0018730, 0.0020904, 0.0018730, 0.0020904, -0.0001177, 0.0001177)
5: (0.0077004, 0.0091132, 0.0077004, 0.0091132, -0.0007649, 0.0007649)
6: (-0.0007722, -0.0004136, -0.0007722, -0.0004136, -0.0001941, 0.0001941)
7: (-0.0051355, -0.0042078, -0.0051355, -0.0042078, -0.0005023, 0.0005023)
8: (-0.0022649, -0.0017770, -0.0022649, -0.0017770, -0.0002641, 0.0002641)
9: (0.0001966, 0.0007624, 0.0001966, 0.0007624, -0.0003063, 0.0003063)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.25 = 2.68 seconds
status: Status.ADV_EXAMPLE
