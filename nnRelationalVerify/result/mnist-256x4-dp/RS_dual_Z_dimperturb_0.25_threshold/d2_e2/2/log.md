## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00149824981


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0000626, 0.0011388, 0.0000626, 0.0011388, -0.0008950, 0.0008950)
1: (0.9934677, 0.9957467, 0.9934677, 0.9957467, -0.0018989, 0.0018989)
2: (-0.0079746, -0.0073425, -0.0079746, -0.0073425, -0.0005169, 0.0005169)
3: (0.0027664, 0.0041127, 0.0027664, 0.0041127, -0.0011223, 0.0011223)
4: (0.0026038, 0.0043586, 0.0026038, 0.0043586, -0.0014973, 0.0014973)
5: (0.0035107, 0.0060608, 0.0035107, 0.0060608, -0.0021106, 0.0021106)
6: (-0.0017774, 0.0005807, -0.0017774, 0.0005807, -0.0019764, 0.0019764)
7: (-0.0078946, -0.0068038, -0.0078946, -0.0068038, -0.0008987, 0.0008987)
8: (0.0078988, 0.0081823, 0.0078988, 0.0081823, -0.0002478, 0.0002478)
9: (-0.0037087, -0.0021511, -0.0037087, -0.0021511, -0.0012945, 0.0012945)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.23 + 1.45 = 2.68 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0012720, upper bound: 0.0012720
