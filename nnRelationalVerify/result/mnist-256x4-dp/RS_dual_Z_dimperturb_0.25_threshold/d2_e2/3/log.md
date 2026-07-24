## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.004636575


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0003460, 0.0003460)
1: (-0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0013232, 0.0013232)
2: (0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0019623, 0.0019623)
3: (-0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0014669, 0.0014669)
4: (-0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0014304, 0.0014304)
5: (0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0014634, 0.0014634)
6: (0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667)
7: (-0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0030852, 0.0030852)
8: (0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0091802, 0.0091802)
9: (0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0026270, 0.0026270)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 1.60 = 2.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0064182, upper bound: 0.0064182

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059465, upper bound: 0.0059465
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059465, upper bound: 0.0059465
time: 0.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 8, lower bound: -0.0059465, upper bound: 0.0059465
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 8, lower bound: -0.0059465, upper bound: 0.0059465

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0003459, 0.0003459
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0013229, 0.0013228
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0019616, 0.0019617
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0014664, 0.0014665
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0014300, 0.0014301
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0014629, 0.0014630
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0030843, 0.0030841
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0091778, 0.0091771
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0026261, 0.0026263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057677, upper bound: 0.0056218
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056218, upper bound: 0.0057677
time: 0.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0003460, 0.0003459
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0013232, 0.0013229
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0019617, 0.0019623
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0014665, 0.0014669
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0014301, 0.0014304
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0014630, 0.0014634
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0030852, 0.0030843
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0091802, 0.0091778
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0026263, 0.0026270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057677, upper bound: 0.0056218
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056218, upper bound: 0.0057677
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 8, lower bound: -0.0057677, upper bound: 0.0056218
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 8, lower bound: -0.0056218, upper bound: 0.0057677
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 8, lower bound: -0.0057677, upper bound: 0.0056218
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 8, lower bound: -0.0056218, upper bound: 0.0057677

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0003340, 0.0003311
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0012771, 0.0012637
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0018686, 0.0018888
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0013943, 0.0014095
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0013805, 0.0013945
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0013908, 0.0014059
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0029456, 0.0029128
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0088422, 0.0087481
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0024849, 0.0025125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051975, upper bound: 0.0052918
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054471, upper bound: 0.0050518
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0003311, 0.0003339
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0012638, 0.0012771
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0018886, 0.0018688
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0014094, 0.0013944
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0013944, 0.0013806
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0014058, 0.0013909
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0029130, 0.0029454
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0087488, 0.0088416
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0025124, 0.0024851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050518, upper bound: 0.0054471
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052918, upper bound: 0.0051975
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0003340, 0.0003311
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0012774, 0.0012638
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0018688, 0.0018893
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0013944, 0.0014099
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0013806, 0.0013948
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0013909, 0.0014063
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0029464, 0.0029130
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0088446, 0.0087488
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0024851, 0.0025132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051975, upper bound: 0.0052918
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054471, upper bound: 0.0050518
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0003312, 0.0003340
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0012641, 0.0012771
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0018888, 0.0018693
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0014095, 0.0013948
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0013945, 0.0013809
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0014059, 0.0013913
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0029138, 0.0029456
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0087512, 0.0088422
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0025125, 0.0024858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050518, upper bound: 0.0054471
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052918, upper bound: 0.0051975
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 8, lower bound: -0.0051975, upper bound: 0.0052918
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 8, lower bound: -0.0054471, upper bound: 0.0050518
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 8, lower bound: -0.0050518, upper bound: 0.0054471
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 8, lower bound: -0.0052918, upper bound: 0.0051975
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 8, lower bound: -0.0051975, upper bound: 0.0052918
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 8, lower bound: -0.0054471, upper bound: 0.0050518
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 8, lower bound: -0.0050518, upper bound: 0.0054471
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 8, lower bound: -0.0052918, upper bound: 0.0051975

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002941, 0.0002990
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011002, 0.0011232
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016501, 0.0016157
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012268, 0.0012008
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012467, 0.0012228
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012232, 0.0011973
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024509, 0.0025071
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075754, 0.0077363
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021569, 0.0021096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051078, upper bound: 0.0052174
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051230, upper bound: 0.0051915
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0003023, 0.0002912
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011387, 0.0010868
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015955, 0.0016734
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011857, 0.0012442
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012088, 0.0012628
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011822, 0.0012406
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0025449, 0.0024181
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0078447, 0.0074814
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020820, 0.0021888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053241, upper bound: 0.0049716
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053752, upper bound: 0.0049695
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002912, 0.0003023
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010869, 0.0011388
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016735, 0.0015957
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012443, 0.0011858
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012629, 0.0012089
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012407, 0.0011823
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024183, 0.0025451
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074820, 0.0078452
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021889, 0.0020822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049695, upper bound: 0.0053752
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049716, upper bound: 0.0053241
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002990, 0.0002940
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011232, 0.0011001
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016156, 0.0016501
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012008, 0.0012267
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012227, 0.0012466
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011972, 0.0012231
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0025069, 0.0024507
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0077359, 0.0075749
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021095, 0.0021568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051915, upper bound: 0.0051230
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052174, upper bound: 0.0051078
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002941, 0.0002990
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011005, 0.0011232
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016501, 0.0016161
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012267, 0.0012012
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012466, 0.0012231
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012231, 0.0011977
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024517, 0.0025069
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075776, 0.0077359
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021568, 0.0021103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051078, upper bound: 0.0052174
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051230, upper bound: 0.0051915
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0003023, 0.0002912
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011391, 0.0010869
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015957, 0.0016738
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011858, 0.0012446
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012089, 0.0012631
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011823, 0.0012410
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0025457, 0.0024183
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0078469, 0.0074820
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020822, 0.0021894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053241, upper bound: 0.0049716
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053752, upper bound: 0.0049695
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002913, 0.0003023
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010872, 0.0011387
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016734, 0.0015961
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012442, 0.0011861
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012628, 0.0012092
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012406, 0.0011827
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024191, 0.0025449
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074842, 0.0078447
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021888, 0.0020828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049695, upper bound: 0.0053752
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049716, upper bound: 0.0053241
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002990, 0.0002941
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011235, 0.0011002
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016157, 0.0016505
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012008, 0.0012270
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012228, 0.0012469
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011973, 0.0012235
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0025077, 0.0024509
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0077381, 0.0075754
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021096, 0.0021574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051915, upper bound: 0.0051230
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052174, upper bound: 0.0051078
time: 0.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0051078, upper bound: 0.0052174
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0051230, upper bound: 0.0051915
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0053241, upper bound: 0.0049716
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0053752, upper bound: 0.0049695
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0049695, upper bound: 0.0053752
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0049716, upper bound: 0.0053241
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0051915, upper bound: 0.0051230
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0052174, upper bound: 0.0051078
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0051078, upper bound: 0.0052174
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0051230, upper bound: 0.0051915
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0053241, upper bound: 0.0049716
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0053752, upper bound: 0.0049695
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0049695, upper bound: 0.0053752
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0049716, upper bound: 0.0053241
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0051915, upper bound: 0.0051230
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 8, lower bound: -0.0052174, upper bound: 0.0051078

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002897, 0.0002958
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010896, 0.0011178
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016416, 0.0015993
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012200, 0.0011882
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012413, 0.0012120
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012164, 0.0011847
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024156, 0.0024845
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074999, 0.0076972
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021407, 0.0020827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049444
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048479, upper bound: 0.0049673
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002905, 0.0002947
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010931, 0.0011126
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016338, 0.0016045
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012141, 0.0011921
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012359, 0.0012156
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012105, 0.0011886
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024242, 0.0024718
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075243, 0.0076608
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021300, 0.0020898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048821, upper bound: 0.0049215
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048738, upper bound: 0.0049427
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002980, 0.0002876
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011281, 0.0010794
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015842, 0.0016570
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011768, 0.0012316
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012015, 0.0012520
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011733, 0.0012280
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0025096, 0.0023909
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0077692, 0.0074291
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020619, 0.0021618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050668, upper bound: 0.0047291
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050564, upper bound: 0.0047364
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002990, 0.0002869
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011329, 0.0010761
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015792, 0.0016642
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011730, 0.0012369
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011980, 0.0012570
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011696, 0.0012333
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0025213, 0.0023828
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0078026, 0.0074058
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020550, 0.0021716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051116, upper bound: 0.0047254
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051039, upper bound: 0.0047356
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002869, 0.0002990
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010762, 0.0011329
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016643, 0.0015793
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012370, 0.0011731
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012570, 0.0011981
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012334, 0.0011697
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023830, 0.0025215
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074065, 0.0078031
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021718, 0.0020552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047356, upper bound: 0.0051039
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047254, upper bound: 0.0051116
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002876, 0.0002980
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010795, 0.0011282
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016571, 0.0015842
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012316, 0.0011768
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012521, 0.0012015
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012280, 0.0011734
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023911, 0.0025098
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074295, 0.0077697
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021620, 0.0020620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047364, upper bound: 0.0050564
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047291, upper bound: 0.0050668
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002946, 0.0002905
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011125, 0.0010930
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016044, 0.0016337
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011920, 0.0012140
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012155, 0.0012358
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011885, 0.0012105
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024717, 0.0024240
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0076604, 0.0075237
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020897, 0.0021298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049427, upper bound: 0.0048738
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049215, upper bound: 0.0048821
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002957, 0.0002897
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011177, 0.0010895
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015992, 0.0016414
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011881, 0.0012199
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012119, 0.0012412
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011846, 0.0012163
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024843, 0.0024154
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0076965, 0.0074993
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020825, 0.0021404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049673, upper bound: 0.0048479
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049444, upper bound: 0.0048668
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002898, 0.0002957
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010899, 0.0011177
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016414, 0.0015998
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012199, 0.0011885
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012412, 0.0012123
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012163, 0.0011850
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024164, 0.0024843
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075021, 0.0076965
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021404, 0.0020833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049444
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048479, upper bound: 0.0049673
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002906, 0.0002946
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010934, 0.0011125
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016337, 0.0016050
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012140, 0.0011925
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012358, 0.0012159
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012105, 0.0011890
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024249, 0.0024717
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075266, 0.0076604
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021298, 0.0020905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048821, upper bound: 0.0049215
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048738, upper bound: 0.0049427
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002980, 0.0002876
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011284, 0.0010795
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015842, 0.0016575
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011768, 0.0012319
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012015, 0.0012523
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011734, 0.0012283
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0025104, 0.0023911
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0077714, 0.0074295
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020620, 0.0021625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050668, upper bound: 0.0047291
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050564, upper bound: 0.0047364
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002991, 0.0002869
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011332, 0.0010762
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015793, 0.0016646
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011731, 0.0012373
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011981, 0.0012573
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011697, 0.0012337
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0025221, 0.0023830
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0078049, 0.0074065
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020552, 0.0021723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051116, upper bound: 0.0047254
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051039, upper bound: 0.0047356
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002870, 0.0002990
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010765, 0.0011329
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016642, 0.0015798
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012369, 0.0011735
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012570, 0.0011984
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012333, 0.0011700
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023838, 0.0025213
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074088, 0.0078026
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021716, 0.0020559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047356, upper bound: 0.0051039
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047254, upper bound: 0.0051116
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002877, 0.0002980
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010798, 0.0011281
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016570, 0.0015847
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012316, 0.0011772
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012520, 0.0012019
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0012280, 0.0011737
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023918, 0.0025096
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074317, 0.0077692
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021618, 0.0020626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047364, upper bound: 0.0050564
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047291, upper bound: 0.0050668
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002947, 0.0002905
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011128, 0.0010931
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016045, 0.0016342
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011921, 0.0012144
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012156, 0.0012362
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011886, 0.0012108
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024724, 0.0024242
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0076627, 0.0075243
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020898, 0.0021305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049427, upper bound: 0.0048738
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049215, upper bound: 0.0048821
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002958, 0.0002897
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011180, 0.0010896
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015993, 0.0016419
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011882, 0.0012202
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012120, 0.0012415
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011847, 0.0012166
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024850, 0.0024156
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0076988, 0.0074999
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020827, 0.0021411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049673, upper bound: 0.0048479
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049444, upper bound: 0.0048668
time: 0.54 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049444
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0048479, upper bound: 0.0049673
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0048821, upper bound: 0.0049215
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0048738, upper bound: 0.0049427
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0050668, upper bound: 0.0047291
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0050564, upper bound: 0.0047364
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0051116, upper bound: 0.0047254
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0051039, upper bound: 0.0047356
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0047356, upper bound: 0.0051039
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0047254, upper bound: 0.0051116
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0047364, upper bound: 0.0050564
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0047291, upper bound: 0.0050668
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0049427, upper bound: 0.0048738
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0049215, upper bound: 0.0048821
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0049673, upper bound: 0.0048479
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0049444, upper bound: 0.0048668
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0048668, upper bound: 0.0049444
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0048479, upper bound: 0.0049673
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0048821, upper bound: 0.0049215
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0048738, upper bound: 0.0049427
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0050668, upper bound: 0.0047291
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0050564, upper bound: 0.0047364
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0051116, upper bound: 0.0047254
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0051039, upper bound: 0.0047356
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0047356, upper bound: 0.0051039
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0047254, upper bound: 0.0051116
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0047364, upper bound: 0.0050564
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0047291, upper bound: 0.0050668
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0049427, upper bound: 0.0048738
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0049215, upper bound: 0.0048821
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0049673, upper bound: 0.0048479
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 8, lower bound: -0.0049444, upper bound: 0.0048668

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002828, 0.0002865
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010588, 0.0010764
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015795, 0.0015533
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011733, 0.0011536
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011983, 0.0011801
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011698, 0.0011501
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023406, 0.0023834
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072849, 0.0074074
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020555, 0.0020195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002805, 0.0002888
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010481, 0.0010870
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015955, 0.0015372
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011853, 0.0011415
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012093, 0.0011689
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011818, 0.0011381
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023145, 0.0024094
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072100, 0.0074820
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020774, 0.0019975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002832, 0.0002854
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010609, 0.0010711
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015717, 0.0015564
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011674, 0.0011559
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011928, 0.0011822
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011639, 0.0011525
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023457, 0.0023706
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072996, 0.0073710
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020448, 0.0020238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002812, 0.0002877
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010516, 0.0010822
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015882, 0.0015425
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011798, 0.0011454
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012043, 0.0011726
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011764, 0.0011420
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023230, 0.0023976
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072345, 0.0074481
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020674, 0.0020047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002907, 0.0002783
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010958, 0.0010380
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015221, 0.0016087
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011301, 0.0011952
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011584, 0.0012185
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011267, 0.0011917
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024309, 0.0022898
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075437, 0.0071393
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0019767, 0.0020955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002887, 0.0002808
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010866, 0.0010498
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015398, 0.0015949
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011434, 0.0011849
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011707, 0.0012089
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011400, 0.0011814
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024085, 0.0023187
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074794, 0.0072221
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020010, 0.0020766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002916, 0.0002776
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011003, 0.0010347
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015171, 0.0016153
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011264, 0.0012002
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011550, 0.0012231
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011230, 0.0011967
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024417, 0.0022817
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075746, 0.0071160
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0019699, 0.0021046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002897, 0.0002803
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010914, 0.0010475
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015362, 0.0016021
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011408, 0.0011903
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011682, 0.0012139
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011373, 0.0011867
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024201, 0.0023129
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075128, 0.0072054
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0019961, 0.0020865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002803, 0.0002897
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010475, 0.0010915
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016022, 0.0015363
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011903, 0.0011408
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012140, 0.0011683
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011868, 0.0011373
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023129, 0.0024203
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072055, 0.0075133
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020866, 0.0019961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002776, 0.0002916
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010348, 0.0011003
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016154, 0.0015172
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012003, 0.0011265
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012232, 0.0011551
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011967, 0.0011231
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0022819, 0.0024419
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0071167, 0.0075750
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021047, 0.0019700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002808, 0.0002887
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010499, 0.0010867
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015950, 0.0015399
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011850, 0.0011435
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012090, 0.0011708
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011815, 0.0011401
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023188, 0.0024087
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072225, 0.0074799
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020768, 0.0020012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002783, 0.0002907
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010381, 0.0010959
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016087, 0.0015222
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011953, 0.0011302
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012185, 0.0011585
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011917, 0.0011268
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0022899, 0.0024310
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0071397, 0.0075439
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020956, 0.0019768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002877, 0.0002812
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010821, 0.0010515
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015423, 0.0015882
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011453, 0.0011798
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011725, 0.0012043
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011419, 0.0011763
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023975, 0.0023228
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074479, 0.0072339
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020045, 0.0020674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002854, 0.0002832
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010711, 0.0010608
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015562, 0.0015716
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011558, 0.0011674
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011821, 0.0011928
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011523, 0.0011639
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023705, 0.0023454
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0073706, 0.0072987
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020235, 0.0020447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002888, 0.0002804
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010870, 0.0010480
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015371, 0.0015954
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011414, 0.0011852
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011689, 0.0012093
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011380, 0.0011817
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024093, 0.0023143
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074816, 0.0072095
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0019973, 0.0020773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002865, 0.0002827
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010762, 0.0010587
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015531, 0.0015794
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011534, 0.0011732
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011799, 0.0011982
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011500, 0.0011697
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023831, 0.0023404
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074067, 0.0072842
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020193, 0.0020553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002828, 0.0002865
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010591, 0.0010762
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015794, 0.0015537
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011732, 0.0011539
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011982, 0.0011804
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011697, 0.0011505
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023414, 0.0023831
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072871, 0.0074067
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020553, 0.0020201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002805, 0.0002888
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010484, 0.0010870
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015954, 0.0015377
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011852, 0.0011419
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012093, 0.0011693
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011817, 0.0011384
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023152, 0.0024093
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072123, 0.0074816
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020773, 0.0019981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002833, 0.0002854
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010612, 0.0010711
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015716, 0.0015569
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011674, 0.0011563
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011928, 0.0011826
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011639, 0.0011528
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023465, 0.0023705
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0073018, 0.0073706
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020447, 0.0020244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002813, 0.0002877
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010519, 0.0010821
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015882, 0.0015429
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011798, 0.0011458
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012043, 0.0011729
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011763, 0.0011424
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023238, 0.0023975
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072367, 0.0074479
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020674, 0.0020053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002907, 0.0002783
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010962, 0.0010381
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015222, 0.0016092
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011302, 0.0011956
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011585, 0.0012188
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011268, 0.0011921
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024317, 0.0022899
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075460, 0.0071397
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0019768, 0.0020962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002888, 0.0002808
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010870, 0.0010499
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015399, 0.0015954
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011435, 0.0011852
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011708, 0.0012093
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011401, 0.0011817
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024092, 0.0023188
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074816, 0.0072225
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020012, 0.0020773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002917, 0.0002776
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0011006, 0.0010348
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015172, 0.0016158
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011265, 0.0012006
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011551, 0.0012234
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011231, 0.0011970
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024425, 0.0022819
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075768, 0.0071167
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0019700, 0.0021053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002898, 0.0002803
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010917, 0.0010475
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015363, 0.0016026
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011408, 0.0011906
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011683, 0.0012142
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011373, 0.0011871
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024209, 0.0023129
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0075151, 0.0072055
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0019961, 0.0020871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002804, 0.0002897
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010478, 0.0010914
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016021, 0.0015367
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011903, 0.0011411
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012139, 0.0011686
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011867, 0.0011377
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023137, 0.0024201
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072078, 0.0075128
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020865, 0.0019968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002777, 0.0002916
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010351, 0.0011003
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016153, 0.0015177
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0012002, 0.0011268
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012231, 0.0011554
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011967, 0.0011234
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0022827, 0.0024417
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0071190, 0.0075746
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0021046, 0.0019707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002809, 0.0002887
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010502, 0.0010866
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015949, 0.0015404
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011849, 0.0011439
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012089, 0.0011711
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011814, 0.0011404
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023196, 0.0024085
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0072248, 0.0074794
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020766, 0.0020018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002784, 0.0002907
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010384, 0.0010958
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0016087, 0.0015226
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011952, 0.0011305
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0012185, 0.0011588
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011917, 0.0011271
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0022907, 0.0024309
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0071419, 0.0075437
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020955, 0.0019774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002878, 0.0002812
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010825, 0.0010516
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015425, 0.0015887
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011454, 0.0011802
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011726, 0.0012046
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011420, 0.0011767
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023983, 0.0023230
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074502, 0.0072345
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020047, 0.0020680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002854, 0.0002832
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010714, 0.0010609
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015564, 0.0015721
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011559, 0.0011677
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011822, 0.0011931
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011525, 0.0011642
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023713, 0.0023457
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0073728, 0.0072996
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020238, 0.0020453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002888, 0.0002805
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010873, 0.0010481
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015372, 0.0015959
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011415, 0.0011856
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011689, 0.0012096
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011381, 0.0011821
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0024100, 0.0023145
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074839, 0.0072100
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0019975, 0.0020779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002910, 0.0001358, -0.0002910, 0.0001358, -0.0002865, 0.0002828
1: -0.0000419, 0.0015298, -0.0000419, 0.0015298, -0.0010766, 0.0010588
2: 0.0140490, 0.0164028, 0.0140490, 0.0164028, -0.0015533, 0.0015798
3: -0.0000627, 0.0017073, -0.0000627, 0.0017073, -0.0011536, 0.0011735
4: -0.0044374, -0.0028048, -0.0044374, -0.0028048, -0.0011801, 0.0011985
5: 0.0078756, 0.0096424, 0.0078756, 0.0096424, -0.0011501, 0.0011700
6: 0.0092942, 0.0099609, 0.0092942, 0.0099609, -0.0006667, 0.0006667
7: -0.0193320, -0.0154965, -0.0193320, -0.0154965, -0.0023839, 0.0023406
8: 0.9684025, 0.9793916, 0.9684025, 0.9793916, -0.0074090, 0.0072849
9: 0.0037100, 0.0069397, 0.0037100, 0.0069397, -0.0020195, 0.0020559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
time: 0.52 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032189, upper bound: 0.0032409
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032462, upper bound: 0.0031976
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033168, upper bound: 0.0031302
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0033504, upper bound: 0.0031014
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031014, upper bound: 0.0033504
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031302, upper bound: 0.0033168
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0031976, upper bound: 0.0032462
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.28
Output dim: 8, lower bound: -0.0032409, upper bound: 0.0032189

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.82 + 243.32 = 246.14 seconds
