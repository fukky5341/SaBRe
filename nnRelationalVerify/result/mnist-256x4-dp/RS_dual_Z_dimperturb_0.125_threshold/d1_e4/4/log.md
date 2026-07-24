## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00167296


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0072934, -0.0042745, -0.0072934, -0.0042745, -0.0011336, 0.0011336)
1: (-0.0047997, -0.0044627, -0.0047997, -0.0044627, -0.0001266, 0.0001266)
2: (0.0355849, 0.0430532, 0.0355849, 0.0430532, -0.0028042, 0.0028042)
3: (0.0025253, 0.0073237, 0.0025253, 0.0073237, -0.0018017, 0.0018017)
4: (-0.0034844, -0.0025944, -0.0034844, -0.0025944, -0.0003342, 0.0003342)
5: (0.0102973, 0.0109186, 0.0102973, 0.0109186, -0.0002333, 0.0002333)
6: (-0.0118904, -0.0048213, -0.0118904, -0.0048213, -0.0026544, 0.0026544)
7: (0.9629830, 0.9716827, 0.9629830, 0.9716827, -0.0032667, 0.0032667)
8: (-0.0045405, -0.0019446, -0.0045405, -0.0019446, -0.0009747, 0.0009747)
9: (-0.0013882, -0.0010980, -0.0013882, -0.0010980, -0.0001090, 0.0001090)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.17 = 2.68 seconds
status: Status.ADV_EXAMPLE
