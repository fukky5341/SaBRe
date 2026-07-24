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
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0071781, -0.0040954, -0.0071781, -0.0040954, -0.0011719, 0.0011719)
1: (-0.0047869, -0.0044427, -0.0047869, -0.0044427, -0.0001308, 0.0001308)
2: (0.0351419, 0.0427680, 0.0351419, 0.0427680, -0.0028990, 0.0028990)
3: (0.0022406, 0.0071405, 0.0022406, 0.0071405, -0.0018626, 0.0018626)
4: (-0.0034504, -0.0025416, -0.0034504, -0.0025416, -0.0003455, 0.0003455)
5: (0.0102605, 0.0108948, 0.0102605, 0.0108948, -0.0002411, 0.0002411)
6: (-0.0116205, -0.0044019, -0.0116205, -0.0044019, -0.0027441, 0.0027441)
7: (0.9633151, 0.9721988, 0.9633151, 0.9721988, -0.0033770, 0.0033770)
8: (-0.0046945, -0.0020437, -0.0046945, -0.0020437, -0.0010077, 0.0010077)
9: (-0.0013771, -0.0010808, -0.0013771, -0.0010808, -0.0001126, 0.0001126)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 1.18 = 2.74 seconds
status: Status.ADV_EXAMPLE
