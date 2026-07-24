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
Threshold: 5.708e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041692, -0.0041632, -0.0041692, -0.0041632, -0.0000029, 0.0000029)
1: (-0.0087916, -0.0085690, -0.0087916, -0.0085690, -0.0001102, 0.0001102)
2: (0.9659132, 0.9661804, 0.9659132, 0.9661804, -0.0001322, 0.0001322)
3: (-0.0051130, -0.0031422, -0.0051130, -0.0031422, -0.0009750, 0.0009750)
4: (-0.0004540, -0.0003042, -0.0004540, -0.0003042, -0.0000742, 0.0000742)
5: (0.0168115, 0.0169629, 0.0168115, 0.0169629, -0.0000749, 0.0000749)
6: (0.0036560, 0.0037297, 0.0036560, 0.0037297, -0.0000365, 0.0000365)
7: (-0.0069639, -0.0064532, -0.0069639, -0.0064532, -0.0002527, 0.0002527)
8: (0.0112043, 0.0116095, 0.0112043, 0.0116095, -0.0002005, 0.0002005)
9: (0.0178766, 0.0186054, 0.0178766, 0.0186054, -0.0003606, 0.0003606)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.16 = 2.52 seconds
status: Status.ADV_EXAMPLE
