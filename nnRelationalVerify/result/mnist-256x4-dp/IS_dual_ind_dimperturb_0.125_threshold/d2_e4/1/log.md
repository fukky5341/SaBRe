## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00086954


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0060592, 0.0083459, 0.0060592, 0.0083459, -0.0012827, 0.0012827)
1: (0.0019372, 0.0041156, 0.0019372, 0.0041156, -0.0012277, 0.0012277)
2: (-0.0203718, -0.0149201, -0.0203718, -0.0149201, -0.0029604, 0.0029604)
3: (-0.0014285, 0.0033024, -0.0014285, 0.0033024, -0.0026340, 0.0026340)
4: (0.0153707, 0.0157730, 0.0153707, 0.0157730, -0.0002771, 0.0002771)
5: (-0.0032610, 0.0034011, -0.0032610, 0.0034011, -0.0037260, 0.0037260)
6: (0.9952555, 0.9997443, 0.9952555, 0.9997443, -0.0024962, 0.0024962)
7: (0.0149752, 0.0169806, 0.0149752, 0.0169806, -0.0010767, 0.0010767)
8: (0.0038259, 0.0058456, 0.0038259, 0.0058456, -0.0011506, 0.0011506)
9: (-0.0230718, -0.0184919, -0.0230718, -0.0184919, -0.0025148, 0.0025148)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.77 + 1.27 = 3.04 seconds
status: Status.ADV_EXAMPLE
