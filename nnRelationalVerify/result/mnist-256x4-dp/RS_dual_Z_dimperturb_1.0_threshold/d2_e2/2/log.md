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
Threshold: 0.08927084141697089


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835)
1: (0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128)
2: (-0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145)
3: (-0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669)
4: (-0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407)
5: (-0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920)
6: (-0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919)
7: (-0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113)
8: (-0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782)
9: (-0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 2.85 = 4.14 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0767329, upper bound: 0.0767329
