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
Threshold: 0.00037578


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042015, -0.0041362, -0.0042015, -0.0041362, -0.0000653, 0.0000653)
1: (-0.0100003, -0.0092136, -0.0100003, -0.0092136, -0.0007867, 0.0007867)
2: (0.9644628, 0.9654068, 0.9644628, 0.9654068, -0.0009440, 0.0009440)
3: (-0.0158111, -0.0088480, -0.0158111, -0.0088480, -0.0053447, 0.0053447)
4: (-0.0000201, 0.0005095, -0.0000201, 0.0005095, -0.0005296, 0.0005296)
5: (0.0172500, 0.0180181, 0.0172500, 0.0180181, -0.0007680, 0.0007680)
6: (0.0026360, 0.0035164, 0.0026360, 0.0035164, -0.0008803, 0.0008803)
7: (-0.0054852, -0.0033125, -0.0054852, -0.0033125, -0.0021727, 0.0021727)
8: (0.0123774, 0.0138091, 0.0123774, 0.0138091, -0.0014316, 0.0014316)
9: (0.0199866, 0.0225616, 0.0199866, 0.0225616, -0.0023945, 0.0023945)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.19 + 1.39 = 2.57 seconds
status: Status.ADV_EXAMPLE
