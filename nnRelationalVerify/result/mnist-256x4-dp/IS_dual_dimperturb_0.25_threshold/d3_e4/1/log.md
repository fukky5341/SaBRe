## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00167283


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0013888, 0.0000306, -0.0013888, 0.0000306, -0.0009899, 0.0009899)
1: (-0.0078348, -0.0042330, -0.0078348, -0.0042330, -0.0025119, 0.0025119)
2: (0.0301693, 0.0324039, 0.0301693, 0.0324039, -0.0015584, 0.0015584)
3: (-0.0009437, 0.0032288, -0.0009437, 0.0032288, -0.0029099, 0.0029099)
4: (-0.0068623, -0.0031987, -0.0068623, -0.0031987, -0.0025550, 0.0025550)
5: (0.0111389, 0.0125266, 0.0111389, 0.0125266, -0.0009678, 0.0009678)
6: (-0.0008095, 0.0044858, -0.0008095, 0.0044858, -0.0036931, 0.0036931)
7: (0.9774928, 0.9811983, 0.9774928, 0.9811983, -0.0025843, 0.0025842)
8: (-0.0106955, -0.0067227, -0.0106955, -0.0067227, -0.0027707, 0.0027707)
9: (-0.0005589, 0.0020654, -0.0005589, 0.0020654, -0.0018302, 0.0018302)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.93 + 1.61 = 3.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0017048, upper bound: 0.0017048

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016827, upper bound: 0.0016159
time: 0.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016827, upper bound: 0.0016827
time: 0.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 7, lower bound: -0.0016827, upper bound: 0.0016159
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 7, lower bound: -0.0016827, upper bound: 0.0016827

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0013496, 0.0000267, -0.0013755, 0.0000293, -0.0009476, 0.0009724
1: -0.0077355, -0.0042429, -0.0078012, -0.0042362, -0.0024048, 0.0024676
2: 0.0302309, 0.0323977, 0.0301901, 0.0324019, -0.0014919, 0.0015309
3: -0.0009323, 0.0031138, -0.0009400, 0.0031899, -0.0028586, 0.0027858
4: -0.0067613, -0.0032087, -0.0068282, -0.0032019, -0.0024461, 0.0025100
5: 0.0111772, 0.0125228, 0.0111519, 0.0125254, -0.0009265, 0.0009507
6: -0.0007951, 0.0043399, -0.0008049, 0.0044365, -0.0036279, 0.0035356
7: 0.9775029, 0.9810961, 0.9774961, 0.9811637, -0.0025386, 0.0024740
8: -0.0106846, -0.0068322, -0.0106920, -0.0067597, -0.0027218, 0.0026525
9: -0.0004866, 0.0020582, -0.0005345, 0.0020631, -0.0017522, 0.0017979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016542, upper bound: 0.0015851
time: 0.69 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016542, upper bound: 0.0015851
time: 0.78 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0013715, 0.0000639, -0.0013804, 0.0000288, -0.0009635, 0.0010204
1: -0.0077909, -0.0041484, -0.0078136, -0.0042374, -0.0024449, 0.0025894
2: 0.0301965, 0.0324563, 0.0301824, 0.0324011, -0.0015168, 0.0016065
3: -0.0010417, 0.0031780, -0.0009386, 0.0032043, -0.0029997, 0.0028323
4: -0.0068177, -0.0031126, -0.0068408, -0.0032031, -0.0024869, 0.0026339
5: 0.0111558, 0.0125592, 0.0111471, 0.0125249, -0.0009420, 0.0009976
6: -0.0009339, 0.0044214, -0.0008031, 0.0044548, -0.0038070, 0.0035946
7: 0.9774057, 0.9811531, 0.9774973, 0.9811766, -0.0026640, 0.0025153
8: -0.0107888, -0.0067710, -0.0106907, -0.0067460, -0.0028562, 0.0026968
9: -0.0005270, 0.0021270, -0.0005435, 0.0020622, -0.0017814, 0.0018867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016542, upper bound: 0.0016542
time: 0.87 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016542, upper bound: 0.0016542
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.46 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 7, lower bound: -0.0016542, upper bound: 0.0015851
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 7, lower bound: -0.0016542, upper bound: 0.0015851
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 7, lower bound: -0.0016542, upper bound: 0.0016542
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 7, lower bound: -0.0016542, upper bound: 0.0016542

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.54 + 8.52 = 12.06 seconds
