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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9877605, 0.9890585, 0.9877605, 0.9890585, -0.0011229, 0.0011229)
1: (-0.0043137, -0.0039903, -0.0043137, -0.0039903, -0.0002798, 0.0002798)
2: (0.0110924, 0.0128063, 0.0110924, 0.0128063, -0.0014828, 0.0014828)
3: (-0.0071020, -0.0063219, -0.0071020, -0.0063219, -0.0006749, 0.0006749)
4: (0.0026748, 0.0030065, 0.0026748, 0.0030065, -0.0002870, 0.0002870)
5: (0.0129108, 0.0150664, 0.0129108, 0.0150664, -0.0018649, 0.0018649)
6: (-0.0022832, -0.0017361, -0.0022832, -0.0017361, -0.0004733, 0.0004733)
7: (-0.0090449, -0.0076294, -0.0090449, -0.0076294, -0.0012247, 0.0012247)
8: (-0.0043208, -0.0035764, -0.0043208, -0.0035764, -0.0006440, 0.0006440)
9: (0.0022831, 0.0031463, 0.0022831, 0.0031463, -0.0007468, 0.0007468)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 1.26 = 2.48 seconds
status: Status.ADV_EXAMPLE
