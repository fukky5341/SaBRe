## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13486788544


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0366059, 0.0212332, -0.0366059, 0.0212332, -0.0578392, 0.0578392)
1: (-0.0342316, 0.0198400, -0.0342316, 0.0198400, -0.0540715, 0.0540715)
2: (-0.0327693, 0.1011313, -0.0327693, 0.1011313, -0.1339006, 0.1339006)
3: (-0.0179999, 0.0443540, -0.0179999, 0.0443540, -0.0623539, 0.0623539)
4: (-0.0369075, 0.0406127, -0.0369075, 0.0406127, -0.0775202, 0.0775202)
5: (-0.0203528, 0.0323051, -0.0203528, 0.0323051, -0.0526579, 0.0526579)
6: (-0.0848213, 0.0472194, -0.0848213, 0.0472194, -0.1320408, 0.1320408)
7: (0.8765109, 1.0003195, 0.8765109, 1.0003195, -0.1238086, 0.1238086)
8: (-0.0659065, 0.0781348, -0.0659065, 0.0781348, -0.1355211, 0.1355211)
9: (-0.0651210, 0.0341468, -0.0651210, 0.0341468, -0.0992679, 0.0992679)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.44 + 1.90 = 3.34 seconds
status: Status.VERIFIED
relational distance
Output dim: 7, lower bound: -0.0995855, upper bound: 0.0995855
