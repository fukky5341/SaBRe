## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00027306


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9946608, 0.9958829, 0.9946608, 0.9958829, -0.0008863, 0.0008863)
1: (-0.0025943, -0.0022898, -0.0025943, -0.0022898, -0.0002208, 0.0002208)
2: (0.0020809, 0.0036946, 0.0020809, 0.0036946, -0.0011703, 0.0011703)
3: (-0.0029548, -0.0022202, -0.0029548, -0.0022202, -0.0005327, 0.0005327)
4: (0.0009306, 0.0012430, 0.0009306, 0.0012430, -0.0002265, 0.0002265)
5: (0.0015766, 0.0036063, 0.0015766, 0.0036063, -0.0014720, 0.0014720)
6: (0.0006255, 0.0011407, 0.0006255, 0.0011407, -0.0003736, 0.0003736)
7: (-0.0015192, -0.0001864, -0.0015192, -0.0001864, -0.0009666, 0.0009666)
8: (-0.0003631, 0.0003378, -0.0003631, 0.0003378, -0.0005083, 0.0005083)
9: (-0.0022556, -0.0014428, -0.0022556, -0.0014428, -0.0005894, 0.0005894)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.01 + 1.25 = 3.26 seconds
status: Status.ADV_EXAMPLE
