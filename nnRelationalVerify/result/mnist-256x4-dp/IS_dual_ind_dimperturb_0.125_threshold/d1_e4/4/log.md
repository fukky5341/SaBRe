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
0: (-0.0072834, -0.0037542, -0.0072834, -0.0037542, -0.0011997, 0.0011997)
1: (-0.0047986, -0.0044046, -0.0047986, -0.0044046, -0.0001339, 0.0001339)
2: (0.0342978, 0.0430284, 0.0342978, 0.0430284, -0.0029679, 0.0029679)
3: (0.0016983, 0.0073078, 0.0016983, 0.0073078, -0.0019069, 0.0019069)
4: (-0.0034814, -0.0024410, -0.0034814, -0.0024410, -0.0003537, 0.0003537)
5: (0.0101903, 0.0109165, 0.0101903, 0.0109165, -0.0002469, 0.0002469)
6: (-0.0118669, -0.0036029, -0.0118669, -0.0036029, -0.0028093, 0.0028093)
7: (0.9630119, 0.9731822, 0.9630119, 0.9731822, -0.0034573, 0.0034573)
8: (-0.0049879, -0.0019532, -0.0049879, -0.0019532, -0.0010316, 0.0010316)
9: (-0.0013872, -0.0010480, -0.0013872, -0.0010480, -0.0001153, 0.0001153)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.55 + 1.16 = 2.71 seconds
status: Status.ADV_EXAMPLE
