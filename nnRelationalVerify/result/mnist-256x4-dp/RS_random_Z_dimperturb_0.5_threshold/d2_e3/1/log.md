## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00027335


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0065989, 0.0072462, 0.0065989, 0.0072462, -0.0003825, 0.0003825)
1: (0.0008636, 0.0021173, 0.0008636, 0.0021173, -0.0007408, 0.0007408)
2: (-0.0001500, 0.0099622, -0.0001500, 0.0099622, -0.0059751, 0.0059751)
3: (-0.0034023, -0.0024992, -0.0034023, -0.0024992, -0.0005337, 0.0005337)
4: (0.0048286, 0.0092106, 0.0048286, 0.0092106, -0.0025892, 0.0025892)
5: (-0.0018947, -0.0012405, -0.0018947, -0.0012405, -0.0003865, 0.0003865)
6: (0.9925284, 0.9937282, 0.9925284, 0.9937282, -0.0007089, 0.0007089)
7: (-0.0046423, 0.0032899, -0.0046423, 0.0032899, -0.0046870, 0.0046870)
8: (-0.0004660, 0.0020191, -0.0004660, 0.0020191, -0.0014684, 0.0014684)
9: (-0.0113589, -0.0063990, -0.0113589, -0.0063990, -0.0029307, 0.0029307)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 1.52 = 2.87 seconds
status: Status.ADV_EXAMPLE
