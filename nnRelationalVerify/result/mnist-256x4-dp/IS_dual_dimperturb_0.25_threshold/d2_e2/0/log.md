## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00011788


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041720, -0.0041575, -0.0041720, -0.0041575, -0.0000093, 0.0000093)
1: (-0.0088963, -0.0083545, -0.0088963, -0.0083545, -0.0003494, 0.0003494)
2: (0.9657875, 0.9664376, 0.9657875, 0.9664376, -0.0004193, 0.0004193)
3: (-0.0060397, -0.0012442, -0.0060397, -0.0012442, -0.0030923, 0.0030923)
4: (-0.0005984, -0.0002337, -0.0005984, -0.0002337, -0.0002352, 0.0002352)
5: (0.0166655, 0.0170342, 0.0166655, 0.0170342, -0.0002377, 0.0002377)
6: (0.0036214, 0.0038007, 0.0036214, 0.0038007, -0.0001156, 0.0001156)
7: (-0.0074558, -0.0062130, -0.0074558, -0.0062130, -0.0008014, 0.0008014)
8: (0.0108141, 0.0118000, 0.0108141, 0.0118000, -0.0006358, 0.0006358)
9: (0.0171747, 0.0189481, 0.0171747, 0.0189481, -0.0011435, 0.0011435)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.20 + 1.23 = 2.43 seconds
status: Status.ADV_EXAMPLE
