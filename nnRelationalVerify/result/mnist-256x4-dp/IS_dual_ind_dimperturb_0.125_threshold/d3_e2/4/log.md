## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0006444


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040647, -0.0039511, -0.0040647, -0.0039511, -0.0000595, 0.0000595)
1: (0.0016254, 0.0022542, 0.0016254, 0.0022542, -0.0003293, 0.0003293)
2: (0.0099300, 0.0113348, 0.0099300, 0.0113348, -0.0007358, 0.0007358)
3: (0.0025578, 0.0031498, 0.0025578, 0.0031498, -0.0003101, 0.0003101)
4: (1.0066737, 1.0089704, 1.0066737, 1.0089704, -0.0012029, 0.0012029)
5: (0.0035397, 0.0039865, 0.0035397, 0.0039865, -0.0002340, 0.0002340)
6: (-0.0109308, -0.0103494, -0.0109308, -0.0103494, -0.0003045, 0.0003045)
7: (-0.0101977, -0.0101235, -0.0101977, -0.0101235, -0.0000388, 0.0000388)
8: (-0.0037035, -0.0033018, -0.0037035, -0.0033018, -0.0002104, 0.0002104)
9: (-0.0016413, 0.0003699, -0.0016413, 0.0003699, -0.0010534, 0.0010534)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 1.21 = 2.87 seconds
status: Status.ADV_EXAMPLE
