## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.001618947


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0027793, 0.0007802, -0.0027793, 0.0007802, -0.0031360, 0.0031360)
1: (-0.0045715, -0.0033847, -0.0045715, -0.0033847, -0.0010818, 0.0010818)
2: (0.0110736, 0.0158771, 0.0110736, 0.0158771, -0.0041543, 0.0041543)
3: (1.0068507, 1.0098901, 1.0068507, 1.0098901, -0.0030394, 0.0030394)
4: (-0.0042237, -0.0034306, -0.0042237, -0.0034306, -0.0006738, 0.0006738)
5: (0.0018269, 0.0045721, 0.0018269, 0.0045721, -0.0024119, 0.0024119)
6: (-0.0025937, -0.0023109, -0.0025937, -0.0023109, -0.0002828, 0.0002828)
7: (-0.0130874, -0.0085539, -0.0130874, -0.0085539, -0.0044894, 0.0044894)
8: (-0.0133740, -0.0047333, -0.0133740, -0.0047333, -0.0072697, 0.0072697)
9: (-0.0018645, 0.0024431, -0.0018645, 0.0024431, -0.0035832, 0.0035832)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 2.35 = 3.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0018723, upper bound: 0.0018724

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018149, upper bound: 0.0018330
time: 1.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018380, upper bound: 0.0018381
time: 1.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.83
Output dim: 3, lower bound: -0.0018149, upper bound: 0.0018330
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.83
Output dim: 3, lower bound: -0.0018380, upper bound: 0.0018381

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0027402, 0.0006512, -0.0027772, 0.0007656, -0.0030338, 0.0029788
1: -0.0045640, -0.0034480, -0.0045711, -0.0033919, -0.0010560, 0.0010156
2: 0.0111187, 0.0156787, 0.0110762, 0.0158547, -0.0040073, 0.0039123
3: 1.0070064, 1.0098710, 1.0068684, 1.0098889, -0.0028825, 0.0030026
4: -0.0041867, -0.0034370, -0.0042195, -0.0034310, -0.0006283, 0.0006477
5: 0.0018565, 0.0044706, 0.0018285, 0.0045607, -0.0023324, 0.0022880
6: -0.0025849, -0.0023124, -0.0025927, -0.0023110, -0.0002739, 0.0002803
7: -0.0130722, -0.0086240, -0.0130857, -0.0085586, -0.0044665, 0.0044127
8: -0.0129440, -0.0047983, -0.0133255, -0.0047369, -0.0067369, 0.0069726
9: -0.0018363, 0.0022133, -0.0018629, 0.0024171, -0.0034335, 0.0032966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018149, upper bound: 0.0018149
time: 1.12 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018149, upper bound: 0.0018330
time: 1.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0027716, 0.0007119, -0.0027793, 0.0007802, -0.0031290, 0.0030222
1: -0.0045702, -0.0034181, -0.0045715, -0.0033847, -0.0010810, 0.0010287
2: 0.0110827, 0.0157721, 0.0110736, 0.0158771, -0.0041460, 0.0039744
3: 1.0069314, 1.0098865, 1.0068507, 1.0098901, -0.0029587, 0.0030358
4: -0.0042041, -0.0034319, -0.0042237, -0.0034306, -0.0006400, 0.0006727
5: 0.0018328, 0.0045184, 0.0018269, 0.0045721, -0.0024065, 0.0023219
6: -0.0025890, -0.0023112, -0.0025937, -0.0023109, -0.0002781, 0.0002826
7: -0.0130794, -0.0085722, -0.0130874, -0.0085539, -0.0044773, 0.0044714
8: -0.0131466, -0.0047459, -0.0133740, -0.0047333, -0.0068758, 0.0072603
9: -0.0018593, 0.0023215, -0.0018645, 0.0024431, -0.0035801, 0.0033717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018330, upper bound: 0.0018149
time: 1.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018330, upper bound: 0.0018381
time: 1.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 3, lower bound: -0.0018149, upper bound: 0.0018149
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 3, lower bound: -0.0018149, upper bound: 0.0018330
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 3, lower bound: -0.0018330, upper bound: 0.0018149
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.41
Output dim: 3, lower bound: -0.0018330, upper bound: 0.0018381

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027402, 0.0006512, -0.0027402, 0.0006512, -0.0028837, 0.0028837
1: -0.0045640, -0.0034480, -0.0045640, -0.0034480, -0.0009880, 0.0009880
2: 0.0111187, 0.0156787, 0.0111187, 0.0156787, -0.0037728, 0.0037728
3: 1.0070064, 1.0098710, 1.0070064, 1.0098710, -0.0028646, 0.0028646
4: -0.0041867, -0.0034370, -0.0041867, -0.0034370, -0.0006031, 0.0006031
5: 0.0018565, 0.0044706, 0.0018565, 0.0044706, -0.0022141, 0.0022141
6: -0.0025849, -0.0023124, -0.0025849, -0.0023124, -0.0002724, 0.0002724
7: -0.0130722, -0.0086240, -0.0130722, -0.0086240, -0.0043951, 0.0043951
8: -0.0129440, -0.0047983, -0.0129440, -0.0047983, -0.0064528, 0.0064528
9: -0.0018363, 0.0022133, -0.0018363, 0.0022133, -0.0031507, 0.0031507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017868, upper bound: 0.0017328
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017920, upper bound: 0.0017929
time: 1.14 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027402, 0.0006512, -0.0027716, 0.0007119, -0.0029993, 0.0029736
1: -0.0045640, -0.0034480, -0.0045702, -0.0034181, -0.0010352, 0.0010151
2: 0.0111187, 0.0156787, 0.0110827, 0.0157721, -0.0039544, 0.0039066
3: 1.0070064, 1.0098710, 1.0069314, 1.0098865, -0.0028801, 0.0029396
4: -0.0041867, -0.0034370, -0.0042041, -0.0034319, -0.0006276, 0.0006379
5: 0.0018565, 0.0044706, 0.0018328, 0.0045184, -0.0023053, 0.0022841
6: -0.0025849, -0.0023124, -0.0025890, -0.0023112, -0.0002737, 0.0002766
7: -0.0130722, -0.0086240, -0.0130794, -0.0085722, -0.0044531, 0.0044087
8: -0.0129440, -0.0047983, -0.0131466, -0.0047459, -0.0067309, 0.0068578
9: -0.0018363, 0.0022133, -0.0018593, 0.0023215, -0.0033722, 0.0032946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017312, upper bound: 0.0018030
time: 1.26 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017920, upper bound: 0.0018091
time: 1.42 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027716, 0.0007119, -0.0027402, 0.0006512, -0.0029736, 0.0029993
1: -0.0045702, -0.0034181, -0.0045640, -0.0034480, -0.0010151, 0.0010352
2: 0.0110827, 0.0157721, 0.0111187, 0.0156787, -0.0039066, 0.0039544
3: 1.0069314, 1.0098865, 1.0070064, 1.0098710, -0.0029396, 0.0028801
4: -0.0042041, -0.0034319, -0.0041867, -0.0034370, -0.0006379, 0.0006276
5: 0.0018328, 0.0045184, 0.0018565, 0.0044706, -0.0022841, 0.0023053
6: -0.0025890, -0.0023112, -0.0025849, -0.0023124, -0.0002766, 0.0002737
7: -0.0130794, -0.0085722, -0.0130722, -0.0086240, -0.0044087, 0.0044531
8: -0.0131466, -0.0047459, -0.0129440, -0.0047983, -0.0068578, 0.0067309
9: -0.0018593, 0.0023215, -0.0018363, 0.0022133, -0.0032946, 0.0033722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018029, upper bound: 0.0017312
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018091, upper bound: 0.0017920
time: 1.46 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027716, 0.0007119, -0.0027716, 0.0007119, -0.0030150, 0.0030150
1: -0.0045702, -0.0034181, -0.0045702, -0.0034181, -0.0010278, 0.0010278
2: 0.0110827, 0.0157721, 0.0110827, 0.0157721, -0.0039659, 0.0039659
3: 1.0069314, 1.0098865, 1.0069314, 1.0098865, -0.0029551, 0.0029551
4: -0.0042041, -0.0034319, -0.0042041, -0.0034319, -0.0006389, 0.0006389
5: 0.0018328, 0.0045184, 0.0018328, 0.0045184, -0.0023165, 0.0023165
6: -0.0025890, -0.0023112, -0.0025890, -0.0023112, -0.0002778, 0.0002778
7: -0.0130794, -0.0085722, -0.0130794, -0.0085722, -0.0044593, 0.0044593
8: -0.0131466, -0.0047459, -0.0131466, -0.0047459, -0.0068663, 0.0068663
9: -0.0018593, 0.0023215, -0.0018593, 0.0023215, -0.0033682, 0.0033682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018029, upper bound: 0.0017364
time: 1.51 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018091, upper bound: 0.0018010
time: 1.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.65 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.0017868, upper bound: 0.0017328
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.0017920, upper bound: 0.0017929
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.0017312, upper bound: 0.0018030
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.0017920, upper bound: 0.0018091
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.0018029, upper bound: 0.0017312
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.0018091, upper bound: 0.0017920
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.0018029, upper bound: 0.0017364
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.0018091, upper bound: 0.0018010

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0025503, 0.0006344, -0.0027090, 0.0006485, -0.0026906, 0.0028127
1: -0.0045162, -0.0034570, -0.0045562, -0.0034494, -0.0009354, 0.0009525
2: 0.0113582, 0.0156529, 0.0111583, 0.0156746, -0.0035307, 0.0036728
3: 1.0070370, 1.0097522, 1.0070115, 1.0098516, -0.0028145, 0.0027406
4: -0.0041819, -0.0034733, -0.0041860, -0.0034430, -0.0005857, 0.0005663
5: 0.0020015, 0.0044574, 0.0018804, 0.0044685, -0.0020664, 0.0021589
6: -0.0025782, -0.0023223, -0.0025836, -0.0023140, -0.0002642, 0.0002614
7: -0.0130703, -0.0089615, -0.0130719, -0.0086798, -0.0043351, 0.0040567
8: -0.0128883, -0.0051725, -0.0129352, -0.0048598, -0.0062529, 0.0060657
9: -0.0016603, 0.0021835, -0.0018076, 0.0022086, -0.0029673, 0.0030471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017544, upper bound: 0.0016899
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017441, upper bound: 0.0016897
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0026862, 0.0006453, -0.0027402, 0.0006512, -0.0028004, 0.0028629
1: -0.0045499, -0.0034511, -0.0045640, -0.0034480, -0.0009769, 0.0009713
2: 0.0111875, 0.0156697, 0.0111187, 0.0156787, -0.0036644, 0.0037409
3: 1.0070170, 1.0098358, 1.0070064, 1.0098710, -0.0028540, 0.0028294
4: -0.0041851, -0.0034475, -0.0041867, -0.0034370, -0.0005972, 0.0005863
5: 0.0018978, 0.0044660, 0.0018565, 0.0044706, -0.0021501, 0.0021977
6: -0.0025825, -0.0023153, -0.0025849, -0.0023124, -0.0002701, 0.0002695
7: -0.0130715, -0.0087282, -0.0130722, -0.0086240, -0.0043927, 0.0042870
8: -0.0129246, -0.0049074, -0.0129440, -0.0047983, -0.0063836, 0.0062717
9: -0.0017843, 0.0022029, -0.0018363, 0.0022133, -0.0030656, 0.0031137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017328, upper bound: 0.0017877
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017328, upper bound: 0.0017929
time: 1.55 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0027090, 0.0006485, -0.0025765, 0.0007042, -0.0029426, 0.0027860
1: -0.0045562, -0.0034494, -0.0045211, -0.0034232, -0.0010024, 0.0009659
2: 0.0111583, 0.0156746, 0.0113279, 0.0157603, -0.0038752, 0.0036715
3: 1.0070115, 1.0098516, 1.0069557, 1.0097642, -0.0027527, 0.0028958
4: -0.0041860, -0.0034430, -0.0042019, -0.0034691, -0.0005920, 0.0006241
5: 0.0018804, 0.0044685, 0.0019817, 0.0045124, -0.0022614, 0.0021408
6: -0.0025836, -0.0023140, -0.0025829, -0.0023213, -0.0002624, 0.0002689
7: -0.0130719, -0.0086798, -0.0130785, -0.0089125, -0.0041142, 0.0043503
8: -0.0129352, -0.0048598, -0.0131209, -0.0051316, -0.0063629, 0.0067020
9: -0.0018076, 0.0022086, -0.0016782, 0.0023078, -0.0032928, 0.0031228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017668
time: 1.24 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017648
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0027402, 0.0006512, -0.0027149, 0.0007057, -0.0029765, 0.0028988
1: -0.0045640, -0.0034480, -0.0045558, -0.0034213, -0.0010172, 0.0010093
2: 0.0111187, 0.0156787, 0.0111539, 0.0157626, -0.0039192, 0.0038110
3: 1.0070064, 1.0098710, 1.0069424, 1.0098507, -0.0028443, 0.0029286
4: -0.0041867, -0.0034370, -0.0042024, -0.0034426, -0.0006131, 0.0006313
5: 0.0018565, 0.0044706, 0.0018761, 0.0045135, -0.0022873, 0.0022268
6: -0.0025849, -0.0023124, -0.0025867, -0.0023141, -0.0002708, 0.0002742
7: -0.0130722, -0.0086240, -0.0130786, -0.0086770, -0.0043435, 0.0044060
8: -0.0129440, -0.0047983, -0.0131259, -0.0048579, -0.0065835, 0.0067816
9: -0.0018363, 0.0022133, -0.0018065, 0.0023105, -0.0033315, 0.0032295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017867, upper bound: 0.0017519
time: 1.43 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017868, upper bound: 0.0018091
time: 1.53 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0025765, 0.0007042, -0.0027090, 0.0006485, -0.0027859, 0.0029426
1: -0.0045211, -0.0034232, -0.0045562, -0.0034494, -0.0009659, 0.0010024
2: 0.0113279, 0.0157603, 0.0111583, 0.0156746, -0.0036715, 0.0038752
3: 1.0069557, 1.0097642, 1.0070115, 1.0098516, -0.0028958, 0.0027527
4: -0.0042019, -0.0034691, -0.0041860, -0.0034430, -0.0006241, 0.0005920
5: 0.0019817, 0.0045124, 0.0018804, 0.0044685, -0.0021408, 0.0022614
6: -0.0025829, -0.0023213, -0.0025836, -0.0023140, -0.0002689, 0.0002624
7: -0.0130785, -0.0089125, -0.0130719, -0.0086798, -0.0043503, 0.0041142
8: -0.0131209, -0.0051316, -0.0129352, -0.0048598, -0.0067020, 0.0063629
9: -0.0016782, 0.0023078, -0.0018076, 0.0022086, -0.0031228, 0.0032928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017668, upper bound: 0.0016881
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017648, upper bound: 0.0016881
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0027149, 0.0007057, -0.0027402, 0.0006512, -0.0028988, 0.0029765
1: -0.0045558, -0.0034213, -0.0045640, -0.0034480, -0.0010093, 0.0010172
2: 0.0111539, 0.0157626, 0.0111187, 0.0156787, -0.0038110, 0.0039192
3: 1.0069424, 1.0098507, 1.0070064, 1.0098710, -0.0029286, 0.0028443
4: -0.0042024, -0.0034426, -0.0041867, -0.0034370, -0.0006313, 0.0006131
5: 0.0018761, 0.0045135, 0.0018565, 0.0044706, -0.0022268, 0.0022873
6: -0.0025867, -0.0023141, -0.0025849, -0.0023124, -0.0002742, 0.0002708
7: -0.0130786, -0.0086770, -0.0130722, -0.0086240, -0.0044060, 0.0043435
8: -0.0131259, -0.0048579, -0.0129440, -0.0047983, -0.0067816, 0.0065835
9: -0.0018065, 0.0023105, -0.0018363, 0.0022133, -0.0032295, 0.0033315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017519, upper bound: 0.0017868
time: 1.56 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017519, upper bound: 0.0017920
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0025765, 0.0007042, -0.0027409, 0.0007092, -0.0028220, 0.0029500
1: -0.0045211, -0.0034232, -0.0045623, -0.0034195, -0.0009741, 0.0009910
2: 0.0113279, 0.0157603, 0.0111215, 0.0157679, -0.0037235, 0.0038744
3: 1.0069557, 1.0097642, 1.0069366, 1.0098671, -0.0029113, 0.0028276
4: -0.0042019, -0.0034691, -0.0042034, -0.0034378, -0.0006228, 0.0006014
5: 0.0019817, 0.0045124, 0.0018562, 0.0045162, -0.0021692, 0.0022661
6: -0.0025829, -0.0023213, -0.0025878, -0.0023128, -0.0002702, 0.0002666
7: -0.0130785, -0.0089125, -0.0130790, -0.0086266, -0.0044009, 0.0041198
8: -0.0131209, -0.0051316, -0.0131374, -0.0048072, -0.0066824, 0.0064698
9: -0.0016782, 0.0023078, -0.0018305, 0.0023166, -0.0031784, 0.0032721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017717, upper bound: 0.0016942
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017691, upper bound: 0.0016942
time: 1.45 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0027149, 0.0007057, -0.0027716, 0.0007119, -0.0029274, 0.0029920
1: -0.0045558, -0.0034213, -0.0045702, -0.0034181, -0.0010101, 0.0010092
2: 0.0111539, 0.0157626, 0.0110827, 0.0157721, -0.0038514, 0.0039307
3: 1.0069424, 1.0098507, 1.0069314, 1.0098865, -0.0029441, 0.0029193
4: -0.0042024, -0.0034426, -0.0042041, -0.0034319, -0.0006323, 0.0006203
5: 0.0018761, 0.0045135, 0.0018328, 0.0045184, -0.0022493, 0.0022984
6: -0.0025867, -0.0023141, -0.0025890, -0.0023112, -0.0002755, 0.0002749
7: -0.0130786, -0.0086770, -0.0130794, -0.0085722, -0.0044566, 0.0043486
8: -0.0131259, -0.0048579, -0.0131466, -0.0047459, -0.0067898, 0.0066642
9: -0.0018065, 0.0023105, -0.0018593, 0.0023215, -0.0032698, 0.0033273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017541, upper bound: 0.0017945
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017541, upper bound: 0.0018010
time: 1.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.64 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017544, upper bound: 0.0016899
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017441, upper bound: 0.0016897
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017328, upper bound: 0.0017877
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017328, upper bound: 0.0017929
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017668
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017648
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017867, upper bound: 0.0017519
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017868, upper bound: 0.0018091
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017668, upper bound: 0.0016881
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017648, upper bound: 0.0016881
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017519, upper bound: 0.0017868
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017519, upper bound: 0.0017920
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017717, upper bound: 0.0016942
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017691, upper bound: 0.0016942
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017541, upper bound: 0.0017945
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 3, lower bound: -0.0017541, upper bound: 0.0018010

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0025488, 0.0005918, -0.0027090, 0.0006485, -0.0026891, 0.0027627
1: -0.0045161, -0.0034773, -0.0045562, -0.0034494, -0.0009353, 0.0009322
2: 0.0113600, 0.0155874, 0.0111583, 0.0156746, -0.0035288, 0.0035961
3: 1.0070835, 1.0097517, 1.0070115, 1.0098516, -0.0027680, 0.0027401
4: -0.0041697, -0.0034735, -0.0041860, -0.0034430, -0.0005714, 0.0005660
5: 0.0020027, 0.0044238, 0.0018804, 0.0044685, -0.0020653, 0.0021196
6: -0.0025761, -0.0023223, -0.0025836, -0.0023140, -0.0002621, 0.0002614
7: -0.0130653, -0.0089641, -0.0130719, -0.0086798, -0.0043292, 0.0040540
8: -0.0127462, -0.0051747, -0.0129352, -0.0048598, -0.0060866, 0.0060633
9: -0.0016595, 0.0021076, -0.0018076, 0.0022086, -0.0029666, 0.0029582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017441, upper bound: 0.0016897
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017441, upper bound: 0.0016897
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0027114, 0.0005442, -0.0027085, 0.0006309, -0.0028818, 0.0027633
1: -0.0045473, -0.0034995, -0.0045561, -0.0034579, -0.0009829, 0.0009481
2: 0.0111683, 0.0155142, 0.0111589, 0.0156476, -0.0037679, 0.0035971
3: 1.0071180, 1.0098295, 1.0070302, 1.0098515, -0.0027335, 0.0027993
4: -0.0041561, -0.0034466, -0.0041810, -0.0034430, -0.0005716, 0.0006018
5: 0.0018798, 0.0043864, 0.0018807, 0.0044547, -0.0022123, 0.0021201
6: -0.0025817, -0.0023159, -0.0025829, -0.0023140, -0.0002677, 0.0002670
7: -0.0130597, -0.0085972, -0.0130699, -0.0086805, -0.0043286, 0.0044263
8: -0.0125876, -0.0049117, -0.0128767, -0.0048605, -0.0060892, 0.0064342
9: -0.0017754, 0.0020228, -0.0018073, 0.0021773, -0.0031445, 0.0029598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016977, upper bound: 0.0016897
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016977, upper bound: 0.0016897
time: 1.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0026862, 0.0006453, -0.0025503, 0.0006344, -0.0027903, 0.0026760
1: -0.0045499, -0.0034511, -0.0045162, -0.0034570, -0.0009482, 0.0009245
2: 0.0111875, 0.0156697, 0.0113582, 0.0156529, -0.0036467, 0.0035083
3: 1.0070170, 1.0098358, 1.0070370, 1.0097522, -0.0027351, 0.0027988
4: -0.0041851, -0.0034475, -0.0041819, -0.0034733, -0.0005621, 0.0005823
5: 0.0018978, 0.0044660, 0.0020015, 0.0044574, -0.0021420, 0.0020550
6: -0.0025825, -0.0023153, -0.0025782, -0.0023223, -0.0002603, 0.0002629
7: -0.0130715, -0.0087282, -0.0130703, -0.0089615, -0.0040550, 0.0042856
8: -0.0129246, -0.0049074, -0.0128883, -0.0051725, -0.0060172, 0.0062197
9: -0.0017843, 0.0022029, -0.0016603, 0.0021835, -0.0030309, 0.0029413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016899, upper bound: 0.0017544
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016897, upper bound: 0.0017441
time: 1.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0026862, 0.0006453, -0.0026862, 0.0006453, -0.0027841, 0.0027841
1: -0.0045499, -0.0034511, -0.0045499, -0.0034511, -0.0009642, 0.0009642
2: 0.0111875, 0.0156697, 0.0111875, 0.0156697, -0.0036392, 0.0036392
3: 1.0070170, 1.0098358, 1.0070170, 1.0098358, -0.0028188, 0.0028188
4: -0.0041851, -0.0034475, -0.0041851, -0.0034475, -0.0005816, 0.0005816
5: 0.0018978, 0.0044660, 0.0018978, 0.0044660, -0.0021372, 0.0021372
6: -0.0025825, -0.0023153, -0.0025825, -0.0023153, -0.0002672, 0.0002672
7: -0.0130715, -0.0087282, -0.0130715, -0.0087282, -0.0042850, 0.0042850
8: -0.0129246, -0.0049074, -0.0129246, -0.0049074, -0.0062171, 0.0062171
9: -0.0017843, 0.0022029, -0.0017843, 0.0022029, -0.0030364, 0.0030364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016961, upper bound: 0.0017494
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016897, upper bound: 0.0017494
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0027090, 0.0006485, -0.0025728, 0.0006592, -0.0028900, 0.0027489
1: -0.0045562, -0.0034494, -0.0045204, -0.0034445, -0.0009803, 0.0009585
2: 0.0111583, 0.0156746, 0.0113323, 0.0156910, -0.0037942, 0.0036121
3: 1.0070115, 1.0098516, 1.0070047, 1.0097625, -0.0027510, 0.0028468
4: -0.0041860, -0.0034430, -0.0041890, -0.0034697, -0.0005809, 0.0006090
5: 0.0018804, 0.0044685, 0.0019845, 0.0044769, -0.0022199, 0.0021117
6: -0.0025836, -0.0023140, -0.0025805, -0.0023214, -0.0002623, 0.0002665
7: -0.0130719, -0.0086798, -0.0130732, -0.0089197, -0.0041039, 0.0043441
8: -0.0129352, -0.0048598, -0.0129709, -0.0051377, -0.0062323, 0.0065265
9: -0.0018076, 0.0022086, -0.0016757, 0.0022276, -0.0031990, 0.0030519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017648
time: 1.42 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017648
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0027085, 0.0006309, -0.0027515, 0.0006175, -0.0028919, 0.0029535
1: -0.0045561, -0.0034579, -0.0045544, -0.0034639, -0.0009946, 0.0010103
2: 0.0111589, 0.0156476, 0.0111212, 0.0156270, -0.0037973, 0.0038701
3: 1.0070302, 1.0098515, 1.0070354, 1.0098472, -0.0028169, 0.0028161
4: -0.0041810, -0.0034430, -0.0041771, -0.0034401, -0.0006202, 0.0006096
5: 0.0018807, 0.0044547, 0.0018494, 0.0044441, -0.0022214, 0.0022680
6: -0.0025829, -0.0023140, -0.0025865, -0.0023144, -0.0002685, 0.0002724
7: -0.0130699, -0.0086805, -0.0130683, -0.0085250, -0.0044994, 0.0043437
8: -0.0128767, -0.0048605, -0.0128321, -0.0048495, -0.0066387, 0.0065337
9: -0.0018073, 0.0021773, -0.0018021, 0.0021535, -0.0032031, 0.0032448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016554, upper bound: 0.0017313
time: 1.36 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016400, upper bound: 0.0017312
time: 1.50 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0025503, 0.0006344, -0.0027149, 0.0007057, -0.0027879, 0.0028795
1: -0.0045162, -0.0034570, -0.0045558, -0.0034213, -0.0009692, 0.0009738
2: 0.0113582, 0.0156529, 0.0111539, 0.0157626, -0.0036821, 0.0037784
3: 1.0070370, 1.0097522, 1.0069424, 1.0098507, -0.0028137, 0.0028098
4: -0.0041819, -0.0034733, -0.0042024, -0.0034426, -0.0006062, 0.0005948
5: 0.0020015, 0.0044574, 0.0018761, 0.0045135, -0.0021433, 0.0022114
6: -0.0025782, -0.0023223, -0.0025867, -0.0023141, -0.0002641, 0.0002644
7: -0.0130703, -0.0089615, -0.0130786, -0.0086770, -0.0043427, 0.0040685
8: -0.0128883, -0.0051725, -0.0131259, -0.0048579, -0.0064925, 0.0064020
9: -0.0016603, 0.0021835, -0.0018065, 0.0023105, -0.0031545, 0.0031734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016958, upper bound: 0.0017124
time: 1.52 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017123
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0026862, 0.0006453, -0.0027149, 0.0007057, -0.0029042, 0.0028833
1: -0.0045499, -0.0034511, -0.0045558, -0.0034213, -0.0010070, 0.0009968
2: 0.0111875, 0.0156697, 0.0111539, 0.0157626, -0.0038263, 0.0037872
3: 1.0070170, 1.0098358, 1.0069424, 1.0098507, -0.0028337, 0.0028934
4: -0.0041851, -0.0034475, -0.0042024, -0.0034426, -0.0006087, 0.0006168
5: 0.0018978, 0.0044660, 0.0018761, 0.0045135, -0.0022318, 0.0022146
6: -0.0025825, -0.0023153, -0.0025867, -0.0023141, -0.0002684, 0.0002713
7: -0.0130715, -0.0087282, -0.0130786, -0.0086770, -0.0043417, 0.0042993
8: -0.0129246, -0.0049074, -0.0131259, -0.0048579, -0.0065320, 0.0066326
9: -0.0017843, 0.0022029, -0.0018065, 0.0023105, -0.0032648, 0.0032019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017730
time: 1.56 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017707
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0025728, 0.0006592, -0.0027090, 0.0006485, -0.0027489, 0.0028900
1: -0.0045204, -0.0034445, -0.0045562, -0.0034494, -0.0009585, 0.0009803
2: 0.0113323, 0.0156910, 0.0111583, 0.0156746, -0.0036121, 0.0037942
3: 1.0070047, 1.0097625, 1.0070115, 1.0098516, -0.0028468, 0.0027510
4: -0.0041890, -0.0034697, -0.0041860, -0.0034430, -0.0006090, 0.0005809
5: 0.0019845, 0.0044769, 0.0018804, 0.0044685, -0.0021117, 0.0022199
6: -0.0025805, -0.0023214, -0.0025836, -0.0023140, -0.0002665, 0.0002623
7: -0.0130732, -0.0089197, -0.0130719, -0.0086798, -0.0043441, 0.0041039
8: -0.0129709, -0.0051377, -0.0129352, -0.0048598, -0.0065265, 0.0062323
9: -0.0016757, 0.0022276, -0.0018076, 0.0022086, -0.0030519, 0.0031990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017648, upper bound: 0.0016881
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017648, upper bound: 0.0016881
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0027515, 0.0006175, -0.0027085, 0.0006309, -0.0029535, 0.0028919
1: -0.0045544, -0.0034639, -0.0045561, -0.0034579, -0.0010103, 0.0009946
2: 0.0111212, 0.0156270, 0.0111589, 0.0156476, -0.0038701, 0.0037973
3: 1.0070354, 1.0098472, 1.0070302, 1.0098515, -0.0028161, 0.0028169
4: -0.0041771, -0.0034401, -0.0041810, -0.0034430, -0.0006096, 0.0006202
5: 0.0018494, 0.0044441, 0.0018807, 0.0044547, -0.0022680, 0.0022214
6: -0.0025865, -0.0023144, -0.0025829, -0.0023140, -0.0002724, 0.0002685
7: -0.0130683, -0.0085250, -0.0130699, -0.0086805, -0.0043437, 0.0044994
8: -0.0128321, -0.0048495, -0.0128767, -0.0048605, -0.0065337, 0.0066387
9: -0.0018021, 0.0021535, -0.0018073, 0.0021773, -0.0032448, 0.0032031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017313, upper bound: 0.0016554
time: 1.49 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017312, upper bound: 0.0016400
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027149, 0.0007057, -0.0025503, 0.0006344, -0.0028795, 0.0027879
1: -0.0045558, -0.0034213, -0.0045162, -0.0034570, -0.0009738, 0.0009692
2: 0.0111539, 0.0157626, 0.0113582, 0.0156529, -0.0037784, 0.0036821
3: 1.0069424, 1.0098507, 1.0070370, 1.0097522, -0.0028098, 0.0028137
4: -0.0042024, -0.0034426, -0.0041819, -0.0034733, -0.0005948, 0.0006062
5: 0.0018761, 0.0045135, 0.0020015, 0.0044574, -0.0022114, 0.0021433
6: -0.0025867, -0.0023141, -0.0025782, -0.0023223, -0.0002644, 0.0002641
7: -0.0130786, -0.0086770, -0.0130703, -0.0089615, -0.0040685, 0.0043427
8: -0.0131259, -0.0048579, -0.0128883, -0.0051725, -0.0064020, 0.0064925
9: -0.0018065, 0.0023105, -0.0016603, 0.0021835, -0.0031734, 0.0031545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017124, upper bound: 0.0017542
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017123, upper bound: 0.0017422
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027149, 0.0007057, -0.0026862, 0.0006453, -0.0028833, 0.0029042
1: -0.0045558, -0.0034213, -0.0045499, -0.0034511, -0.0009968, 0.0010070
2: 0.0111539, 0.0157626, 0.0111875, 0.0156697, -0.0037872, 0.0038263
3: 1.0069424, 1.0098507, 1.0070170, 1.0098358, -0.0028934, 0.0028337
4: -0.0042024, -0.0034426, -0.0041851, -0.0034475, -0.0006168, 0.0006087
5: 0.0018761, 0.0045135, 0.0018978, 0.0044660, -0.0022146, 0.0022318
6: -0.0025867, -0.0023141, -0.0025825, -0.0023153, -0.0002713, 0.0002684
7: -0.0130786, -0.0086770, -0.0130715, -0.0087282, -0.0042993, 0.0043417
8: -0.0131259, -0.0048579, -0.0129246, -0.0049074, -0.0066326, 0.0065320
9: -0.0018065, 0.0023105, -0.0017843, 0.0022029, -0.0032019, 0.0032648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017140, upper bound: 0.0017474
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017123, upper bound: 0.0017474
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0025728, 0.0006592, -0.0027409, 0.0007092, -0.0027819, 0.0028961
1: -0.0045204, -0.0034445, -0.0045623, -0.0034195, -0.0009719, 0.0009705
2: 0.0113323, 0.0156910, 0.0111215, 0.0157679, -0.0036627, 0.0037915
3: 1.0070047, 1.0097625, 1.0069366, 1.0098671, -0.0028623, 0.0028260
4: -0.0041890, -0.0034697, -0.0042034, -0.0034378, -0.0006073, 0.0005899
5: 0.0019845, 0.0044769, 0.0018562, 0.0045162, -0.0021376, 0.0022237
6: -0.0025805, -0.0023214, -0.0025878, -0.0023128, -0.0002678, 0.0002664
7: -0.0130732, -0.0089197, -0.0130790, -0.0086266, -0.0043946, 0.0041094
8: -0.0129709, -0.0051377, -0.0131374, -0.0048072, -0.0065028, 0.0063349
9: -0.0016757, 0.0022276, -0.0018305, 0.0023166, -0.0031114, 0.0031762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017691, upper bound: 0.0016942
time: 1.65 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017691, upper bound: 0.0016942
time: 1.66 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0027515, 0.0006175, -0.0027397, 0.0006926, -0.0029789, 0.0028973
1: -0.0045544, -0.0034639, -0.0045621, -0.0034277, -0.0010192, 0.0009845
2: 0.0111212, 0.0156270, 0.0111229, 0.0157424, -0.0039075, 0.0037938
3: 1.0070354, 1.0098472, 1.0069547, 1.0098665, -0.0028311, 0.0028925
4: -0.0041771, -0.0034401, -0.0041986, -0.0034380, -0.0006078, 0.0006265
5: 0.0018494, 0.0044441, 0.0018571, 0.0045032, -0.0022878, 0.0022247
6: -0.0025865, -0.0023144, -0.0025870, -0.0023128, -0.0002736, 0.0002726
7: -0.0130683, -0.0085250, -0.0130771, -0.0086288, -0.0043927, 0.0045038
8: -0.0128321, -0.0048495, -0.0130821, -0.0048093, -0.0065088, 0.0067168
9: -0.0018021, 0.0021535, -0.0018296, 0.0022871, -0.0032883, 0.0031797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017212, upper bound: 0.0016942
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017212, upper bound: 0.0016942
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027149, 0.0007057, -0.0025765, 0.0007042, -0.0029230, 0.0028052
1: -0.0045558, -0.0034213, -0.0045211, -0.0034232, -0.0009849, 0.0009607
2: 0.0111539, 0.0157626, 0.0113279, 0.0157603, -0.0038423, 0.0036976
3: 1.0069424, 1.0098507, 1.0069557, 1.0097642, -0.0028218, 0.0028950
4: -0.0042024, -0.0034426, -0.0042019, -0.0034691, -0.0005966, 0.0006179
5: 0.0018761, 0.0045135, 0.0019817, 0.0045124, -0.0022457, 0.0021560
6: -0.0025867, -0.0023141, -0.0025829, -0.0023213, -0.0002654, 0.0002688
7: -0.0130786, -0.0086770, -0.0130785, -0.0089125, -0.0041179, 0.0043488
8: -0.0131259, -0.0048579, -0.0131209, -0.0051316, -0.0064138, 0.0066338
9: -0.0018065, 0.0023105, -0.0016782, 0.0023078, -0.0032498, 0.0031485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017137, upper bound: 0.0017623
time: 1.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017137, upper bound: 0.0017514
time: 1.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027149, 0.0007057, -0.0027149, 0.0007057, -0.0029097, 0.0029097
1: -0.0045558, -0.0034213, -0.0045558, -0.0034213, -0.0009964, 0.0009964
2: 0.0111539, 0.0157626, 0.0111539, 0.0157626, -0.0038242, 0.0038242
3: 1.0069424, 1.0098507, 1.0069424, 1.0098507, -0.0029083, 0.0029083
4: -0.0042024, -0.0034426, -0.0042024, -0.0034426, -0.0006153, 0.0006153
5: 0.0018761, 0.0045135, 0.0018761, 0.0045135, -0.0022354, 0.0022354
6: -0.0025867, -0.0023141, -0.0025867, -0.0023141, -0.0002726, 0.0002726
7: -0.0130786, -0.0086770, -0.0130786, -0.0086770, -0.0043465, 0.0043465
8: -0.0131259, -0.0048579, -0.0131259, -0.0048579, -0.0066054, 0.0066054
9: -0.0018065, 0.0023105, -0.0018065, 0.0023105, -0.0032384, 0.0032384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017156, upper bound: 0.0017591
time: 1.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017137, upper bound: 0.0017591
time: 1.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.88 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017441, upper bound: 0.0016897
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017441, upper bound: 0.0016897
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016977, upper bound: 0.0016897
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016977, upper bound: 0.0016897
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016899, upper bound: 0.0017544
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016897, upper bound: 0.0017441
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016961, upper bound: 0.0017494
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016897, upper bound: 0.0017494
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017648
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017648
IS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016554, upper bound: 0.0017313
IS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016400, upper bound: 0.0017312
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016958, upper bound: 0.0017124
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017123
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017730
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0016881, upper bound: 0.0017707
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017648, upper bound: 0.0016881
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017648, upper bound: 0.0016881
IS_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017313, upper bound: 0.0016554
IS_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017312, upper bound: 0.0016400
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017124, upper bound: 0.0017542
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017123, upper bound: 0.0017422
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017140, upper bound: 0.0017474
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017123, upper bound: 0.0017474
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017691, upper bound: 0.0016942
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017691, upper bound: 0.0016942
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017212, upper bound: 0.0016942
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017212, upper bound: 0.0016942
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017137, upper bound: 0.0017623
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017137, upper bound: 0.0017514
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017156, upper bound: 0.0017591
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.88
Output dim: 3, lower bound: -0.0017137, upper bound: 0.0017591

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0025488, 0.0005918, -0.0027076, 0.0006082, -0.0026440, 0.0027613
1: -0.0045161, -0.0034773, -0.0045560, -0.0034688, -0.0009157, 0.0009320
2: 0.0113600, 0.0155874, 0.0111600, 0.0156127, -0.0034595, 0.0035944
3: 1.0070835, 1.0097517, 1.0070580, 1.0098512, -0.0027677, 0.0026937
4: -0.0041697, -0.0034735, -0.0041745, -0.0034432, -0.0005711, 0.0005531
5: 0.0020027, 0.0044238, 0.0018815, 0.0044368, -0.0020299, 0.0021186
6: -0.0025761, -0.0023223, -0.0025817, -0.0023141, -0.0002620, 0.0002594
7: -0.0130653, -0.0089641, -0.0130672, -0.0086821, -0.0043269, 0.0040488
8: -0.0127462, -0.0051747, -0.0128011, -0.0048620, -0.0060843, 0.0059131
9: -0.0016595, 0.0021076, -0.0018069, 0.0021369, -0.0028863, 0.0029575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017234, upper bound: 0.0016571
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017234, upper bound: 0.0016432
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0025488, 0.0005918, -0.0028679, 0.0005541, -0.0026356, 0.0029567
1: -0.0045161, -0.0034773, -0.0045864, -0.0034941, -0.0009141, 0.0009857
2: 0.0113600, 0.0155874, 0.0109718, 0.0155295, -0.0034466, 0.0038437
3: 1.0070835, 1.0097517, 1.0070959, 1.0099272, -0.0028436, 0.0026557
4: -0.0041697, -0.0034735, -0.0041590, -0.0034167, -0.0006099, 0.0005507
5: 0.0020027, 0.0044238, 0.0017601, 0.0043942, -0.0020232, 0.0022678
6: -0.0025761, -0.0023223, -0.0025873, -0.0023078, -0.0002683, 0.0002650
7: -0.0130653, -0.0089641, -0.0130608, -0.0083257, -0.0046889, 0.0040478
8: -0.0127462, -0.0051747, -0.0126208, -0.0046046, -0.0064932, 0.0058850
9: -0.0016595, 0.0021076, -0.0019199, 0.0020406, -0.0028713, 0.0031553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017234, upper bound: 0.0016571
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017234, upper bound: 0.0016432
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027114, 0.0005442, -0.0025498, 0.0006176, -0.0028492, 0.0026078
1: -0.0045473, -0.0034995, -0.0045162, -0.0034652, -0.0009632, 0.0009104
2: 0.0111683, 0.0155142, 0.0113588, 0.0156271, -0.0037177, 0.0034036
3: 1.0071180, 1.0098295, 1.0070552, 1.0097520, -0.0026340, 0.0027744
4: -0.0041561, -0.0034466, -0.0041771, -0.0034734, -0.0005426, 0.0005924
5: 0.0018798, 0.0043864, 0.0020019, 0.0044442, -0.0021866, 0.0020013
6: -0.0025817, -0.0023159, -0.0025774, -0.0023223, -0.0002594, 0.0002616
7: -0.0130597, -0.0085972, -0.0130683, -0.0089622, -0.0040462, 0.0044224
8: -0.0125876, -0.0049117, -0.0128323, -0.0051732, -0.0057906, 0.0063253
9: -0.0017754, 0.0020228, -0.0016600, 0.0021536, -0.0030863, 0.0028205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016603, upper bound: 0.0016404
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016433, upper bound: 0.0016403
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027114, 0.0005442, -0.0026857, 0.0006278, -0.0028673, 0.0027410
1: -0.0045473, -0.0034995, -0.0045498, -0.0034595, -0.0009719, 0.0009438
2: 0.0111683, 0.0155142, 0.0111880, 0.0156427, -0.0037456, 0.0035710
3: 1.0071180, 1.0098295, 1.0070359, 1.0098357, -0.0027177, 0.0027937
4: -0.0041561, -0.0034466, -0.0041800, -0.0034476, -0.0005682, 0.0005976
5: 0.0018798, 0.0043864, 0.0018981, 0.0044522, -0.0022009, 0.0021031
6: -0.0025817, -0.0023159, -0.0025818, -0.0023153, -0.0002664, 0.0002659
7: -0.0130597, -0.0085972, -0.0130695, -0.0087289, -0.0042792, 0.0044246
8: -0.0125876, -0.0049117, -0.0128662, -0.0049082, -0.0060560, 0.0063859
9: -0.0017754, 0.0020228, -0.0017841, 0.0021717, -0.0031187, 0.0029436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016603, upper bound: 0.0016404
time: 1.77 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016433, upper bound: 0.0016403
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0026862, 0.0006453, -0.0025488, 0.0005918, -0.0027404, 0.0026746
1: -0.0045499, -0.0034511, -0.0045161, -0.0034773, -0.0009278, 0.0009244
2: 0.0111875, 0.0156697, 0.0113600, 0.0155874, -0.0035699, 0.0035065
3: 1.0070170, 1.0098358, 1.0070835, 1.0097517, -0.0027347, 0.0027523
4: -0.0041851, -0.0034475, -0.0041697, -0.0034735, -0.0005619, 0.0005680
5: 0.0018978, 0.0044660, 0.0020027, 0.0044238, -0.0021027, 0.0020539
6: -0.0025825, -0.0023153, -0.0025761, -0.0023223, -0.0002602, 0.0002608
7: -0.0130715, -0.0087282, -0.0130653, -0.0089641, -0.0040523, 0.0042797
8: -0.0129246, -0.0049074, -0.0127462, -0.0051747, -0.0060148, 0.0060534
9: -0.0017843, 0.0022029, -0.0016595, 0.0021076, -0.0029420, 0.0029407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016898, upper bound: 0.0017441
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016898, upper bound: 0.0017441
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0026857, 0.0006278, -0.0027114, 0.0005442, -0.0027410, 0.0028673
1: -0.0045498, -0.0034595, -0.0045473, -0.0034995, -0.0009438, 0.0009719
2: 0.0111880, 0.0156427, 0.0111683, 0.0155142, -0.0035710, 0.0037456
3: 1.0070359, 1.0098357, 1.0071180, 1.0098295, -0.0027937, 0.0027177
4: -0.0041800, -0.0034476, -0.0041561, -0.0034466, -0.0005976, 0.0005682
5: 0.0018981, 0.0044522, 0.0018798, 0.0043864, -0.0021031, 0.0022009
6: -0.0025818, -0.0023153, -0.0025817, -0.0023159, -0.0002659, 0.0002664
7: -0.0130695, -0.0087289, -0.0130597, -0.0085972, -0.0044246, 0.0042792
8: -0.0128662, -0.0049082, -0.0125876, -0.0049117, -0.0063859, 0.0060560
9: -0.0017841, 0.0021717, -0.0017754, 0.0020228, -0.0029436, 0.0031187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016404, upper bound: 0.0017134
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016403, upper bound: 0.0017088
time: 1.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0026848, 0.0006050, -0.0026862, 0.0006453, -0.0027826, 0.0027340
1: -0.0045497, -0.0034705, -0.0045499, -0.0034511, -0.0009641, 0.0009432
2: 0.0111892, 0.0156078, 0.0111875, 0.0156697, -0.0036374, 0.0035622
3: 1.0070637, 1.0098355, 1.0070170, 1.0098358, -0.0027721, 0.0028185
4: -0.0041735, -0.0034477, -0.0041851, -0.0034475, -0.0005673, 0.0005813
5: 0.0018989, 0.0044343, 0.0018978, 0.0044660, -0.0021361, 0.0020978
6: -0.0025806, -0.0023154, -0.0025825, -0.0023153, -0.0002652, 0.0002672
7: -0.0130668, -0.0087306, -0.0130715, -0.0087282, -0.0042792, 0.0042827
8: -0.0127904, -0.0049096, -0.0129246, -0.0049074, -0.0060504, 0.0062147
9: -0.0017836, 0.0021312, -0.0017843, 0.0022029, -0.0030358, 0.0029473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017067, upper bound: 0.0017494
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017067, upper bound: 0.0017494
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028438, 0.0005509, -0.0026857, 0.0006278, -0.0029713, 0.0027286
1: -0.0045801, -0.0034958, -0.0045498, -0.0034595, -0.0010124, 0.0009560
2: 0.0110026, 0.0155245, 0.0111880, 0.0156427, -0.0038751, 0.0035540
3: 1.0071018, 1.0099113, 1.0070359, 1.0098357, -0.0027339, 0.0028754
4: -0.0041580, -0.0034214, -0.0041800, -0.0034476, -0.0005657, 0.0006171
5: 0.0017787, 0.0043916, 0.0018981, 0.0044522, -0.0022802, 0.0020936
6: -0.0025861, -0.0023091, -0.0025818, -0.0023153, -0.0002708, 0.0002727
7: -0.0130605, -0.0083745, -0.0130695, -0.0087289, -0.0042779, 0.0046413
8: -0.0126099, -0.0046524, -0.0128662, -0.0049082, -0.0060330, 0.0065866
9: -0.0018965, 0.0020347, -0.0017841, 0.0021717, -0.0032155, 0.0029382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016795, upper bound: 0.0017176
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016718, upper bound: 0.0017176
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0027076, 0.0006082, -0.0025728, 0.0006592, -0.0028885, 0.0027038
1: -0.0045560, -0.0034688, -0.0045204, -0.0034445, -0.0009802, 0.0009389
2: 0.0111600, 0.0156127, 0.0113323, 0.0156910, -0.0037925, 0.0035428
3: 1.0070580, 1.0098512, 1.0070047, 1.0097625, -0.0027045, 0.0028465
4: -0.0041745, -0.0034432, -0.0041890, -0.0034697, -0.0005680, 0.0006088
5: 0.0018815, 0.0044368, 0.0019845, 0.0044769, -0.0022188, 0.0020762
6: -0.0025817, -0.0023141, -0.0025805, -0.0023214, -0.0002603, 0.0002665
7: -0.0130672, -0.0086821, -0.0130732, -0.0089197, -0.0040986, 0.0043418
8: -0.0128011, -0.0048620, -0.0129709, -0.0051377, -0.0060821, 0.0065241
9: -0.0018069, 0.0021369, -0.0016757, 0.0022276, -0.0031983, 0.0029716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016560, upper bound: 0.0017360
time: 1.33 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016426, upper bound: 0.0017360
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028679, 0.0005541, -0.0025728, 0.0006592, -0.0030829, 0.0026954
1: -0.0045864, -0.0034941, -0.0045204, -0.0034445, -0.0010332, 0.0009373
2: 0.0109718, 0.0155295, 0.0113323, 0.0156910, -0.0040396, 0.0035298
3: 1.0070959, 1.0099272, 1.0070047, 1.0097625, -0.0026666, 0.0029224
4: -0.0041590, -0.0034167, -0.0041890, -0.0034697, -0.0005656, 0.0006471
5: 0.0017601, 0.0043942, 0.0019845, 0.0044769, -0.0023675, 0.0020696
6: -0.0025873, -0.0023078, -0.0025805, -0.0023214, -0.0002659, 0.0002727
7: -0.0130608, -0.0083257, -0.0130732, -0.0089197, -0.0040976, 0.0047035
8: -0.0126208, -0.0046046, -0.0129709, -0.0051377, -0.0060540, 0.0069347
9: -0.0019199, 0.0020406, -0.0016757, 0.0022276, -0.0033937, 0.0029566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016560, upper bound: 0.0017360
time: 1.17 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016426, upper bound: 0.0017360
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0027085, 0.0006309, -0.0027134, 0.0006075, -0.0028837, 0.0029164
1: -0.0045561, -0.0034579, -0.0045452, -0.0034689, -0.0009894, 0.0010004
2: 0.0111589, 0.0156476, 0.0111685, 0.0156116, -0.0037847, 0.0038231
3: 1.0070302, 1.0098515, 1.0070500, 1.0098244, -0.0027941, 0.0028014
4: -0.0041810, -0.0034430, -0.0041742, -0.0034472, -0.0006129, 0.0006073
5: 0.0018807, 0.0044547, 0.0018784, 0.0044362, -0.0022149, 0.0022396
6: -0.0025829, -0.0023140, -0.0025846, -0.0023163, -0.0002666, 0.0002706
7: -0.0130699, -0.0086805, -0.0130671, -0.0085906, -0.0044345, 0.0043428
8: -0.0128767, -0.0048605, -0.0127986, -0.0049214, -0.0065618, 0.0065063
9: -0.0018073, 0.0021773, -0.0017681, 0.0021356, -0.0031884, 0.0032085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016554, upper bound: 0.0016693
time: 1.46 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016554, upper bound: 0.0017313
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0026940, 0.0006291, -0.0026691, 0.0006386, -0.0029305, 0.0028952
1: -0.0045522, -0.0034588, -0.0045308, -0.0034533, -0.0010190, 0.0009980
2: 0.0111775, 0.0156448, 0.0112277, 0.0156594, -0.0038601, 0.0038005
3: 1.0070331, 1.0098417, 1.0070105, 1.0097885, -0.0027554, 0.0028312
4: -0.0041804, -0.0034460, -0.0041832, -0.0034571, -0.0006103, 0.0006219
5: 0.0018919, 0.0044532, 0.0019126, 0.0044607, -0.0022522, 0.0022238
6: -0.0025824, -0.0023149, -0.0025863, -0.0023193, -0.0002631, 0.0002714
7: -0.0130696, -0.0087050, -0.0130708, -0.0086514, -0.0043749, 0.0043251
8: -0.0128706, -0.0048910, -0.0129023, -0.0050283, -0.0065402, 0.0066818
9: -0.0017928, 0.0021740, -0.0017153, 0.0021910, -0.0032839, 0.0032015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016400, upper bound: 0.0016692
time: 1.41 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016400, upper bound: 0.0017312
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0025488, 0.0005918, -0.0027149, 0.0007057, -0.0027864, 0.0028280
1: -0.0045161, -0.0034773, -0.0045558, -0.0034213, -0.0009691, 0.0009541
2: 0.0113600, 0.0155874, 0.0111539, 0.0157626, -0.0036802, 0.0036993
3: 1.0070835, 1.0097517, 1.0069424, 1.0098507, -0.0027672, 0.0028093
4: -0.0041697, -0.0034735, -0.0042024, -0.0034426, -0.0005915, 0.0005945
5: 0.0020027, 0.0044238, 0.0018761, 0.0045135, -0.0021420, 0.0021709
6: -0.0025761, -0.0023223, -0.0025867, -0.0023141, -0.0002620, 0.0002644
7: -0.0130653, -0.0089641, -0.0130786, -0.0086770, -0.0043366, 0.0040659
8: -0.0127462, -0.0051747, -0.0131259, -0.0048579, -0.0063210, 0.0063993
9: -0.0016595, 0.0021076, -0.0018065, 0.0023105, -0.0031538, 0.0030818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017422, upper bound: 0.0017119
time: 1.46 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017422, upper bound: 0.0017123
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0027114, 0.0005442, -0.0027136, 0.0006891, -0.0029756, 0.0028248
1: -0.0045473, -0.0034995, -0.0045556, -0.0034295, -0.0010153, 0.0009665
2: 0.0111683, 0.0155142, 0.0111553, 0.0157371, -0.0039167, 0.0036946
3: 1.0071180, 1.0098295, 1.0069607, 1.0098501, -0.0027322, 0.0028688
4: -0.0041561, -0.0034466, -0.0041976, -0.0034428, -0.0005907, 0.0006301
5: 0.0018798, 0.0043864, 0.0018770, 0.0045005, -0.0022864, 0.0021683
6: -0.0025817, -0.0023159, -0.0025859, -0.0023142, -0.0002676, 0.0002700
7: -0.0130597, -0.0085972, -0.0130767, -0.0086793, -0.0043341, 0.0044374
8: -0.0125876, -0.0049117, -0.0130706, -0.0048600, -0.0063121, 0.0067710
9: -0.0017754, 0.0020228, -0.0018057, 0.0022809, -0.0033267, 0.0030773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017127, upper bound: 0.0016650
time: 1.24 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017084, upper bound: 0.0016649
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0026862, 0.0006453, -0.0027117, 0.0006629, -0.0028516, 0.0028459
1: -0.0045499, -0.0034511, -0.0045552, -0.0034419, -0.0009847, 0.0009895
2: 0.0111875, 0.0156697, 0.0111577, 0.0156968, -0.0037454, 0.0037274
3: 1.0070170, 1.0098358, 1.0069911, 1.0098493, -0.0028323, 0.0028447
4: -0.0041851, -0.0034475, -0.0041901, -0.0034432, -0.0005975, 0.0006017
5: 0.0018978, 0.0044660, 0.0018785, 0.0044798, -0.0021904, 0.0021852
6: -0.0025825, -0.0023153, -0.0025845, -0.0023142, -0.0002683, 0.0002692
7: -0.0130715, -0.0087282, -0.0130736, -0.0086833, -0.0043323, 0.0042931
8: -0.0129246, -0.0049074, -0.0129833, -0.0048635, -0.0064009, 0.0064573
9: -0.0017843, 0.0022029, -0.0018043, 0.0022343, -0.0031711, 0.0031297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017046, upper bound: 0.0017707
time: 1.45 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017046, upper bound: 0.0017707
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0026857, 0.0006278, -0.0029294, 0.0006169, -0.0028484, 0.0030943
1: -0.0045498, -0.0034595, -0.0045988, -0.0034638, -0.0010000, 0.0010567
2: 0.0111880, 0.0156427, 0.0108971, 0.0156260, -0.0037407, 0.0040445
3: 1.0070359, 1.0098357, 1.0070214, 1.0099580, -0.0029222, 0.0028143
4: -0.0041800, -0.0034476, -0.0041769, -0.0034060, -0.0006466, 0.0006008
5: 0.0018981, 0.0044522, 0.0017135, 0.0044436, -0.0021879, 0.0023753
6: -0.0025818, -0.0023153, -0.0025917, -0.0023053, -0.0002765, 0.0002763
7: -0.0130695, -0.0087289, -0.0130682, -0.0082207, -0.0047975, 0.0042921
8: -0.0128662, -0.0049082, -0.0128299, -0.0044966, -0.0069153, 0.0064474
9: -0.0017841, 0.0021717, -0.0019662, 0.0021523, -0.0031661, 0.0033781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016707, upper bound: 0.0017421
time: 1.37 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016707, upper bound: 0.0017400
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0025728, 0.0006592, -0.0027076, 0.0006082, -0.0027038, 0.0028885
1: -0.0045204, -0.0034445, -0.0045560, -0.0034688, -0.0009389, 0.0009802
2: 0.0113323, 0.0156910, 0.0111600, 0.0156127, -0.0035428, 0.0037925
3: 1.0070047, 1.0097625, 1.0070580, 1.0098512, -0.0028465, 0.0027045
4: -0.0041890, -0.0034697, -0.0041745, -0.0034432, -0.0006088, 0.0005680
5: 0.0019845, 0.0044769, 0.0018815, 0.0044368, -0.0020762, 0.0022188
6: -0.0025805, -0.0023214, -0.0025817, -0.0023141, -0.0002665, 0.0002603
7: -0.0130732, -0.0089197, -0.0130672, -0.0086821, -0.0043418, 0.0040986
8: -0.0129709, -0.0051377, -0.0128011, -0.0048620, -0.0065241, 0.0060821
9: -0.0016757, 0.0022276, -0.0018069, 0.0021369, -0.0029716, 0.0031983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017360, upper bound: 0.0016560
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017360, upper bound: 0.0016426
time: 1.57 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0025728, 0.0006592, -0.0028679, 0.0005541, -0.0026954, 0.0030829
1: -0.0045204, -0.0034445, -0.0045864, -0.0034941, -0.0009373, 0.0010332
2: 0.0113323, 0.0156910, 0.0109718, 0.0155295, -0.0035298, 0.0040396
3: 1.0070047, 1.0097625, 1.0070959, 1.0099272, -0.0029224, 0.0026666
4: -0.0041890, -0.0034697, -0.0041590, -0.0034167, -0.0006471, 0.0005656
5: 0.0019845, 0.0044769, 0.0017601, 0.0043942, -0.0020696, 0.0023675
6: -0.0025805, -0.0023214, -0.0025873, -0.0023078, -0.0002727, 0.0002659
7: -0.0130732, -0.0089197, -0.0130608, -0.0083257, -0.0047035, 0.0040976
8: -0.0129709, -0.0051377, -0.0126208, -0.0046046, -0.0069347, 0.0060540
9: -0.0016757, 0.0022276, -0.0019199, 0.0020406, -0.0029566, 0.0033937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017360, upper bound: 0.0016560
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017360, upper bound: 0.0016426
time: 1.48 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0027134, 0.0006075, -0.0027085, 0.0006309, -0.0029164, 0.0028837
1: -0.0045452, -0.0034689, -0.0045561, -0.0034579, -0.0010004, 0.0009894
2: 0.0111685, 0.0156116, 0.0111589, 0.0156476, -0.0038231, 0.0037847
3: 1.0070500, 1.0098244, 1.0070302, 1.0098515, -0.0028014, 0.0027941
4: -0.0041742, -0.0034472, -0.0041810, -0.0034430, -0.0006073, 0.0006129
5: 0.0018784, 0.0044362, 0.0018807, 0.0044547, -0.0022396, 0.0022149
6: -0.0025846, -0.0023163, -0.0025829, -0.0023140, -0.0002706, 0.0002666
7: -0.0130671, -0.0085906, -0.0130699, -0.0086805, -0.0043428, 0.0044345
8: -0.0127986, -0.0049214, -0.0128767, -0.0048605, -0.0065063, 0.0065618
9: -0.0017681, 0.0021356, -0.0018073, 0.0021773, -0.0032085, 0.0031884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016693, upper bound: 0.0016554
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016693, upper bound: 0.0016554
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0026691, 0.0006386, -0.0026940, 0.0006291, -0.0028952, 0.0029305
1: -0.0045308, -0.0034533, -0.0045522, -0.0034588, -0.0009980, 0.0010190
2: 0.0112277, 0.0156594, 0.0111775, 0.0156448, -0.0038005, 0.0038601
3: 1.0070105, 1.0097885, 1.0070331, 1.0098417, -0.0028312, 0.0027554
4: -0.0041832, -0.0034571, -0.0041804, -0.0034460, -0.0006219, 0.0006103
5: 0.0019126, 0.0044607, 0.0018919, 0.0044532, -0.0022238, 0.0022522
6: -0.0025863, -0.0023193, -0.0025824, -0.0023149, -0.0002714, 0.0002631
7: -0.0130708, -0.0086514, -0.0130696, -0.0087050, -0.0043251, 0.0043749
8: -0.0129023, -0.0050283, -0.0128706, -0.0048910, -0.0066818, 0.0065402
9: -0.0017153, 0.0021910, -0.0017928, 0.0021740, -0.0032015, 0.0032839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016692, upper bound: 0.0016400
time: 1.47 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016692, upper bound: 0.0016400
time: 1.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0027149, 0.0007057, -0.0025488, 0.0005918, -0.0028280, 0.0027864
1: -0.0045558, -0.0034213, -0.0045161, -0.0034773, -0.0009541, 0.0009691
2: 0.0111539, 0.0157626, 0.0113600, 0.0155874, -0.0036993, 0.0036802
3: 1.0069424, 1.0098507, 1.0070835, 1.0097517, -0.0028093, 0.0027672
4: -0.0042024, -0.0034426, -0.0041697, -0.0034735, -0.0005945, 0.0005915
5: 0.0018761, 0.0045135, 0.0020027, 0.0044238, -0.0021709, 0.0021420
6: -0.0025867, -0.0023141, -0.0025761, -0.0023223, -0.0002644, 0.0002620
7: -0.0130786, -0.0086770, -0.0130653, -0.0089641, -0.0040659, 0.0043366
8: -0.0131259, -0.0048579, -0.0127462, -0.0051747, -0.0063993, 0.0063210
9: -0.0018065, 0.0023105, -0.0016595, 0.0021076, -0.0030818, 0.0031538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017119, upper bound: 0.0017421
time: 1.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017119, upper bound: 0.0017422
time: 1.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0027136, 0.0006891, -0.0027114, 0.0005442, -0.0028248, 0.0029756
1: -0.0045556, -0.0034295, -0.0045473, -0.0034995, -0.0009665, 0.0010153
2: 0.0111553, 0.0157371, 0.0111683, 0.0155142, -0.0036946, 0.0039167
3: 1.0069607, 1.0098501, 1.0071180, 1.0098295, -0.0028688, 0.0027322
4: -0.0041976, -0.0034428, -0.0041561, -0.0034466, -0.0006301, 0.0005907
5: 0.0018770, 0.0045005, 0.0018798, 0.0043864, -0.0021683, 0.0022864
6: -0.0025859, -0.0023142, -0.0025817, -0.0023159, -0.0002700, 0.0002676
7: -0.0130767, -0.0086793, -0.0130597, -0.0085972, -0.0044374, 0.0043341
8: -0.0130706, -0.0048600, -0.0125876, -0.0049117, -0.0067710, 0.0063121
9: -0.0018057, 0.0022809, -0.0017754, 0.0020228, -0.0030773, 0.0033267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016649, upper bound: 0.0017127
time: 1.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016649, upper bound: 0.0017083
time: 1.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027117, 0.0006629, -0.0026862, 0.0006453, -0.0028459, 0.0028516
1: -0.0045552, -0.0034419, -0.0045499, -0.0034511, -0.0009895, 0.0009847
2: 0.0111577, 0.0156968, 0.0111875, 0.0156697, -0.0037274, 0.0037454
3: 1.0069911, 1.0098493, 1.0070170, 1.0098358, -0.0028447, 0.0028323
4: -0.0041901, -0.0034432, -0.0041851, -0.0034475, -0.0006017, 0.0005975
5: 0.0018785, 0.0044798, 0.0018978, 0.0044660, -0.0021852, 0.0021904
6: -0.0025845, -0.0023142, -0.0025825, -0.0023153, -0.0002692, 0.0002683
7: -0.0130736, -0.0086833, -0.0130715, -0.0087282, -0.0042931, 0.0043323
8: -0.0129833, -0.0048635, -0.0129246, -0.0049074, -0.0064573, 0.0064009
9: -0.0018043, 0.0022343, -0.0017843, 0.0022029, -0.0031297, 0.0031711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017288, upper bound: 0.0017474
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017288, upper bound: 0.0017474
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0029294, 0.0006169, -0.0026857, 0.0006278, -0.0030943, 0.0028484
1: -0.0045988, -0.0034638, -0.0045498, -0.0034595, -0.0010567, 0.0010000
2: 0.0108971, 0.0156260, 0.0111880, 0.0156427, -0.0040445, 0.0037407
3: 1.0070214, 1.0099580, 1.0070359, 1.0098357, -0.0028143, 0.0029222
4: -0.0041769, -0.0034060, -0.0041800, -0.0034476, -0.0006008, 0.0006466
5: 0.0017135, 0.0044436, 0.0018981, 0.0044522, -0.0023753, 0.0021879
6: -0.0025917, -0.0023053, -0.0025818, -0.0023153, -0.0002763, 0.0002765
7: -0.0130682, -0.0082207, -0.0130695, -0.0087289, -0.0042921, 0.0047975
8: -0.0128299, -0.0044966, -0.0128662, -0.0049082, -0.0064474, 0.0069153
9: -0.0019662, 0.0021523, -0.0017841, 0.0021717, -0.0033781, 0.0031661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016795, upper bound: 0.0017164
time: 1.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016953, upper bound: 0.0017164
time: 1.54 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0025728, 0.0006592, -0.0027377, 0.0006664, -0.0027380, 0.0028542
1: -0.0045204, -0.0034445, -0.0045617, -0.0034401, -0.0009515, 0.0009685
2: 0.0113323, 0.0156910, 0.0111253, 0.0157021, -0.0035952, 0.0037284
3: 1.0070047, 1.0097625, 1.0069853, 1.0098656, -0.0028609, 0.0027772
4: -0.0041890, -0.0034697, -0.0041911, -0.0034384, -0.0005951, 0.0005773
5: 0.0019845, 0.0044769, 0.0018587, 0.0044825, -0.0021030, 0.0021907
6: -0.0025805, -0.0023214, -0.0025857, -0.0023129, -0.0002677, 0.0002643
7: -0.0130732, -0.0089197, -0.0130740, -0.0086327, -0.0043851, 0.0041043
8: -0.0129709, -0.0051377, -0.0129947, -0.0048128, -0.0063586, 0.0061885
9: -0.0016757, 0.0022276, -0.0018282, 0.0022404, -0.0030332, 0.0031036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017397, upper bound: 0.0016612
time: 1.45 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017397, upper bound: 0.0016480
time: 1.46 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0025728, 0.0006592, -0.0029551, 0.0006203, -0.0027240, 0.0030999
1: -0.0045204, -0.0034445, -0.0046055, -0.0034619, -0.0009536, 0.0010346
2: 0.0113323, 0.0156910, 0.0108646, 0.0156313, -0.0035737, 0.0040379
3: 1.0070047, 1.0097625, 1.0070151, 1.0099747, -0.0029700, 0.0027474
4: -0.0041890, -0.0034697, -0.0041779, -0.0034009, -0.0006438, 0.0005733
5: 0.0019845, 0.0044769, 0.0016938, 0.0044463, -0.0020920, 0.0023785
6: -0.0025805, -0.0023214, -0.0025929, -0.0023039, -0.0002767, 0.0002715
7: -0.0130732, -0.0089197, -0.0130686, -0.0081669, -0.0048530, 0.0041026
8: -0.0129709, -0.0051377, -0.0128413, -0.0044440, -0.0068720, 0.0061421
9: -0.0016757, 0.0022276, -0.0019910, 0.0021584, -0.0030084, 0.0033474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017397, upper bound: 0.0016612
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017397, upper bound: 0.0016480
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027515, 0.0006175, -0.0025753, 0.0006884, -0.0029439, 0.0027406
1: -0.0045544, -0.0034639, -0.0045209, -0.0034310, -0.0009979, 0.0009453
2: 0.0111212, 0.0156270, 0.0113293, 0.0157359, -0.0038537, 0.0035987
3: 1.0070354, 1.0098472, 1.0069737, 1.0097637, -0.0027283, 0.0028734
4: -0.0041771, -0.0034401, -0.0041974, -0.0034693, -0.0005782, 0.0006165
5: 0.0018494, 0.0044441, 0.0019826, 0.0044999, -0.0022602, 0.0021052
6: -0.0025865, -0.0023144, -0.0025821, -0.0023213, -0.0002652, 0.0002677
7: -0.0130683, -0.0085250, -0.0130766, -0.0089148, -0.0041082, 0.0044996
8: -0.0128321, -0.0048495, -0.0130681, -0.0051336, -0.0062005, 0.0066002
9: -0.0018021, 0.0021535, -0.0016774, 0.0022796, -0.0032260, 0.0030348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016846, upper bound: 0.0016467
time: 1.69 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016693, upper bound: 0.0016467
time: 1.56 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027515, 0.0006175, -0.0027136, 0.0006891, -0.0029605, 0.0028703
1: -0.0045544, -0.0034639, -0.0045556, -0.0034295, -0.0010056, 0.0009784
2: 0.0111212, 0.0156270, 0.0111553, 0.0157371, -0.0038792, 0.0037617
3: 1.0070354, 1.0098472, 1.0069607, 1.0098501, -0.0028148, 0.0028864
4: -0.0041771, -0.0034401, -0.0041976, -0.0034428, -0.0006030, 0.0006212
5: 0.0018494, 0.0044441, 0.0018770, 0.0045005, -0.0022733, 0.0022042
6: -0.0025865, -0.0023144, -0.0025859, -0.0023142, -0.0002723, 0.0002715
7: -0.0130683, -0.0085250, -0.0130767, -0.0086793, -0.0043405, 0.0045016
8: -0.0128321, -0.0048495, -0.0130706, -0.0048600, -0.0064603, 0.0066554
9: -0.0018021, 0.0021535, -0.0018057, 0.0022809, -0.0032555, 0.0031573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016694, upper bound: 0.0016610
time: 1.51 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016693, upper bound: 0.0016467
time: 1.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0027149, 0.0007057, -0.0025728, 0.0006592, -0.0028691, 0.0027636
1: -0.0045558, -0.0034213, -0.0045204, -0.0034445, -0.0009644, 0.0009582
2: 0.0111539, 0.0157626, 0.0113323, 0.0156910, -0.0037594, 0.0036345
3: 1.0069424, 1.0098507, 1.0070047, 1.0097625, -0.0028201, 0.0028460
4: -0.0042024, -0.0034426, -0.0041890, -0.0034697, -0.0005846, 0.0006025
5: 0.0018761, 0.0045135, 0.0019845, 0.0044769, -0.0022032, 0.0021231
6: -0.0025867, -0.0023141, -0.0025805, -0.0023214, -0.0002653, 0.0002664
7: -0.0130786, -0.0086770, -0.0130732, -0.0089197, -0.0041073, 0.0043425
8: -0.0131259, -0.0048579, -0.0129709, -0.0051377, -0.0062737, 0.0064543
9: -0.0018065, 0.0023105, -0.0016757, 0.0022276, -0.0031538, 0.0030787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017134, upper bound: 0.0017514
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017134, upper bound: 0.0017514
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0027136, 0.0006891, -0.0027515, 0.0006175, -0.0028703, 0.0029605
1: -0.0045556, -0.0034295, -0.0045544, -0.0034639, -0.0009784, 0.0010056
2: 0.0111553, 0.0157371, 0.0111212, 0.0156270, -0.0037617, 0.0038792
3: 1.0069607, 1.0098501, 1.0070354, 1.0098472, -0.0028864, 0.0028148
4: -0.0041976, -0.0034428, -0.0041771, -0.0034401, -0.0006212, 0.0006030
5: 0.0018770, 0.0045005, 0.0018494, 0.0044441, -0.0022042, 0.0022733
6: -0.0025859, -0.0023142, -0.0025865, -0.0023144, -0.0002715, 0.0002723
7: -0.0130767, -0.0086793, -0.0130683, -0.0085250, -0.0045016, 0.0043405
8: -0.0130706, -0.0048600, -0.0128321, -0.0048495, -0.0066554, 0.0064603
9: -0.0018057, 0.0022809, -0.0018021, 0.0021535, -0.0031573, 0.0032555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016796, upper bound: 0.0017173
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016652, upper bound: 0.0017174
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027117, 0.0006629, -0.0027149, 0.0007057, -0.0028646, 0.0028572
1: -0.0045552, -0.0034419, -0.0045558, -0.0034213, -0.0009940, 0.0009764
2: 0.0111577, 0.0156968, 0.0111539, 0.0157626, -0.0037548, 0.0037435
3: 1.0069911, 1.0098493, 1.0069424, 1.0098507, -0.0028596, 0.0029069
4: -0.0041901, -0.0034432, -0.0042024, -0.0034426, -0.0006002, 0.0006022
5: 0.0018785, 0.0044798, 0.0018761, 0.0045135, -0.0021997, 0.0021941
6: -0.0025845, -0.0023142, -0.0025867, -0.0023141, -0.0002704, 0.0002724
7: -0.0130736, -0.0086833, -0.0130786, -0.0086770, -0.0043404, 0.0043367
8: -0.0129833, -0.0048635, -0.0131259, -0.0048579, -0.0064305, 0.0064523
9: -0.0018043, 0.0022343, -0.0018065, 0.0023105, -0.0031605, 0.0031450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017347, upper bound: 0.0017591
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017347, upper bound: 0.0017591
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0029294, 0.0006169, -0.0027136, 0.0006891, -0.0031050, 0.0028540
1: -0.0045988, -0.0034638, -0.0045556, -0.0034295, -0.0010562, 0.0009922
2: 0.0108971, 0.0156260, 0.0111553, 0.0157371, -0.0040579, 0.0037389
3: 1.0070214, 1.0099580, 1.0069607, 1.0098501, -0.0028287, 0.0029973
4: -0.0041769, -0.0034060, -0.0041976, -0.0034428, -0.0005994, 0.0006485
5: 0.0017135, 0.0044436, 0.0018770, 0.0045005, -0.0023833, 0.0021915
6: -0.0025917, -0.0023053, -0.0025859, -0.0023142, -0.0002775, 0.0002806
7: -0.0130682, -0.0082207, -0.0130767, -0.0086793, -0.0043379, 0.0048007
8: -0.0128299, -0.0044966, -0.0130706, -0.0048600, -0.0064216, 0.0069392
9: -0.0019662, 0.0021523, -0.0018057, 0.0022809, -0.0033907, 0.0031405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017094, upper bound: 0.0017286
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016999, upper bound: 0.0017286
time: 1.21 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.11 seconds
IS_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017234, upper bound: 0.0016571
IS_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017234, upper bound: 0.0016432
IS_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017234, upper bound: 0.0016571
IS_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017234, upper bound: 0.0016432
IS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016603, upper bound: 0.0016404
IS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016433, upper bound: 0.0016403
IS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016603, upper bound: 0.0016404
IS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016433, upper bound: 0.0016403
IS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016898, upper bound: 0.0017441
IS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016898, upper bound: 0.0017441
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016404, upper bound: 0.0017134
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016403, upper bound: 0.0017088
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017067, upper bound: 0.0017494
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017067, upper bound: 0.0017494
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016795, upper bound: 0.0017176
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016718, upper bound: 0.0017176
IS_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016560, upper bound: 0.0017360
IS_A1_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016426, upper bound: 0.0017360
IS_A1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016560, upper bound: 0.0017360
IS_A1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016426, upper bound: 0.0017360
IS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016554, upper bound: 0.0016693
IS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016554, upper bound: 0.0017313
IS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016400, upper bound: 0.0016692
IS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016400, upper bound: 0.0017312
IS_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017422, upper bound: 0.0017119
IS_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017422, upper bound: 0.0017123
IS_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017127, upper bound: 0.0016650
IS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017084, upper bound: 0.0016649
IS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017046, upper bound: 0.0017707
IS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017046, upper bound: 0.0017707
IS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016707, upper bound: 0.0017421
IS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016707, upper bound: 0.0017400
IS_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017360, upper bound: 0.0016560
IS_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017360, upper bound: 0.0016426
IS_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017360, upper bound: 0.0016560
IS_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017360, upper bound: 0.0016426
IS_A2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016693, upper bound: 0.0016554
IS_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016693, upper bound: 0.0016554
IS_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016692, upper bound: 0.0016400
IS_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016692, upper bound: 0.0016400
IS_A2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017119, upper bound: 0.0017421
IS_A2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017119, upper bound: 0.0017422
IS_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016649, upper bound: 0.0017127
IS_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016649, upper bound: 0.0017083
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017288, upper bound: 0.0017474
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017288, upper bound: 0.0017474
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016795, upper bound: 0.0017164
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016953, upper bound: 0.0017164
IS_A2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017397, upper bound: 0.0016612
IS_A2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017397, upper bound: 0.0016480
IS_A2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017397, upper bound: 0.0016612
IS_A2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017397, upper bound: 0.0016480
IS_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016846, upper bound: 0.0016467
IS_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016693, upper bound: 0.0016467
IS_A2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016694, upper bound: 0.0016610
IS_A2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016693, upper bound: 0.0016467
IS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017134, upper bound: 0.0017514
IS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017134, upper bound: 0.0017514
IS_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016796, upper bound: 0.0017173
IS_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016652, upper bound: 0.0017174
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017347, upper bound: 0.0017591
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017347, upper bound: 0.0017591
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0017094, upper bound: 0.0017286
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 3, lower bound: -0.0016999, upper bound: 0.0017286

## BFS IS instance: IS_A1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0025141, 0.0005828, -0.0027076, 0.0006082, -0.0026071, 0.0027546
1: -0.0045068, -0.0034818, -0.0045560, -0.0034688, -0.0009045, 0.0009276
2: 0.0114038, 0.0155736, 0.0111600, 0.0156127, -0.0034101, 0.0035841
3: 1.0070962, 1.0097286, 1.0070580, 1.0098512, -0.0027550, 0.0026705
4: -0.0041672, -0.0034804, -0.0041745, -0.0034432, -0.0005692, 0.0005453
5: 0.0020292, 0.0044168, 0.0018815, 0.0044368, -0.0020014, 0.0021133
6: -0.0025745, -0.0023242, -0.0025817, -0.0023141, -0.0002605, 0.0002575
7: -0.0130642, -0.0090211, -0.0130672, -0.0086821, -0.0043261, 0.0039919
8: -0.0127163, -0.0052474, -0.0128011, -0.0048620, -0.0060618, 0.0058286
9: -0.0016253, 0.0020916, -0.0018069, 0.0021369, -0.0028454, 0.0029455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016866, upper bound: 0.0016807
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016866, upper bound: 0.0016807
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0024596, 0.0006193, -0.0026930, 0.0006064, -0.0025829, 0.0028171
1: -0.0044914, -0.0034635, -0.0045521, -0.0034698, -0.0009022, 0.0009604
2: 0.0114749, 0.0156297, 0.0111786, 0.0156099, -0.0033858, 0.0036838
3: 1.0070525, 1.0096903, 1.0070612, 1.0098413, -0.0027888, 0.0026290
4: -0.0041776, -0.0034915, -0.0041739, -0.0034461, -0.0005884, 0.0005425
5: 0.0020709, 0.0044455, 0.0018926, 0.0044353, -0.0019835, 0.0021627
6: -0.0025762, -0.0023274, -0.0025811, -0.0023149, -0.0002613, 0.0002538
7: -0.0130685, -0.0091083, -0.0130670, -0.0087066, -0.0043103, 0.0039098
8: -0.0128378, -0.0053646, -0.0127949, -0.0048925, -0.0062898, 0.0058061
9: -0.0015689, 0.0021565, -0.0017923, 0.0021336, -0.0028383, 0.0030692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016866, upper bound: 0.0016729
time: 1.58 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016866, upper bound: 0.0016729
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0025141, 0.0005828, -0.0028679, 0.0005541, -0.0025987, 0.0029499
1: -0.0045068, -0.0034818, -0.0045864, -0.0034941, -0.0009029, 0.0009813
2: 0.0114038, 0.0155736, 0.0109718, 0.0155295, -0.0033971, 0.0038333
3: 1.0070962, 1.0097286, 1.0070959, 1.0099272, -0.0028310, 0.0026326
4: -0.0041672, -0.0034804, -0.0041590, -0.0034167, -0.0006080, 0.0005429
5: 0.0020292, 0.0044168, 0.0017601, 0.0043942, -0.0019948, 0.0022625
6: -0.0025745, -0.0023242, -0.0025873, -0.0023078, -0.0002667, 0.0002631
7: -0.0130642, -0.0090211, -0.0130608, -0.0083257, -0.0046881, 0.0039909
8: -0.0127163, -0.0052474, -0.0126208, -0.0046046, -0.0064707, 0.0058005
9: -0.0016253, 0.0020916, -0.0019199, 0.0020406, -0.0028303, 0.0031433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016909, upper bound: 0.0015996
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016790, upper bound: 0.0015982
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0024596, 0.0006193, -0.0028540, 0.0005523, -0.0025744, 0.0030122
1: -0.0044914, -0.0034635, -0.0045826, -0.0034951, -0.0009006, 0.0010137
2: 0.0114749, 0.0156297, 0.0109897, 0.0155267, -0.0033728, 0.0039327
3: 1.0070525, 1.0096903, 1.0070990, 1.0099176, -0.0028651, 0.0025913
4: -0.0041776, -0.0034915, -0.0041584, -0.0034195, -0.0006271, 0.0005401
5: 0.0020709, 0.0044455, 0.0017708, 0.0043927, -0.0019768, 0.0023118
6: -0.0025762, -0.0023274, -0.0025868, -0.0023086, -0.0002676, 0.0002594
7: -0.0130685, -0.0091083, -0.0130606, -0.0083486, -0.0046734, 0.0039088
8: -0.0128378, -0.0053646, -0.0126146, -0.0046337, -0.0066982, 0.0057780
9: -0.0015689, 0.0021565, -0.0019060, 0.0020373, -0.0028233, 0.0032655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016904, upper bound: 0.0015794
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016780, upper bound: 0.0015775
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0027114, 0.0005442, -0.0025152, 0.0006087, -0.0028424, 0.0025708
1: -0.0045473, -0.0034995, -0.0045069, -0.0034697, -0.0009588, 0.0008992
2: 0.0111683, 0.0155142, 0.0114026, 0.0156133, -0.0037073, 0.0033541
3: 1.0071180, 1.0098295, 1.0070678, 1.0097288, -0.0026108, 0.0027617
4: -0.0041561, -0.0034466, -0.0041746, -0.0034803, -0.0005349, 0.0005905
5: 0.0018798, 0.0043864, 0.0020284, 0.0044371, -0.0021813, 0.0019729
6: -0.0025817, -0.0023159, -0.0025759, -0.0023242, -0.0002575, 0.0002600
7: -0.0130597, -0.0085972, -0.0130672, -0.0090193, -0.0039894, 0.0044216
8: -0.0125876, -0.0049117, -0.0128024, -0.0052459, -0.0057061, 0.0063027
9: -0.0017754, 0.0020228, -0.0016258, 0.0021376, -0.0030743, 0.0027795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015932, upper bound: 0.0015945
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015881, upper bound: 0.0015684
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0026973, 0.0005423, -0.0024608, 0.0006445, -0.0029007, 0.0025453
1: -0.0045433, -0.0035005, -0.0044915, -0.0034517, -0.0009907, 0.0008962
2: 0.0111866, 0.0155113, 0.0114736, 0.0156684, -0.0038005, 0.0033278
3: 1.0071212, 1.0098197, 1.0070260, 1.0096906, -0.0025694, 0.0027938
4: -0.0041556, -0.0034495, -0.0041848, -0.0034914, -0.0005317, 0.0006084
5: 0.0018906, 0.0043849, 0.0020701, 0.0044653, -0.0022275, 0.0019539
6: -0.0025812, -0.0023167, -0.0025776, -0.0023273, -0.0002539, 0.0002609
7: -0.0130595, -0.0086202, -0.0130714, -0.0091061, -0.0039074, 0.0044070
8: -0.0125813, -0.0049419, -0.0129218, -0.0053631, -0.0056791, 0.0065164
9: -0.0017610, 0.0020194, -0.0015693, 0.0022014, -0.0031897, 0.0027700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015719, upper bound: 0.0015931
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0015656
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0027114, 0.0005442, -0.0026506, 0.0006190, -0.0028611, 0.0027010
1: -0.0045473, -0.0034995, -0.0045406, -0.0034639, -0.0009676, 0.0009314
2: 0.0111683, 0.0155142, 0.0112330, 0.0156293, -0.0037360, 0.0035204
3: 1.0071180, 1.0098295, 1.0070486, 1.0098127, -0.0026947, 0.0027809
4: -0.0041561, -0.0034466, -0.0041775, -0.0034545, -0.0005601, 0.0005958
5: 0.0018798, 0.0043864, 0.0019250, 0.0044453, -0.0021960, 0.0020724
6: -0.0025817, -0.0023159, -0.0025801, -0.0023173, -0.0002645, 0.0002643
7: -0.0130597, -0.0085972, -0.0130685, -0.0087912, -0.0042165, 0.0044238
8: -0.0125876, -0.0049117, -0.0128369, -0.0049803, -0.0059670, 0.0063651
9: -0.0017754, 0.0020228, -0.0017500, 0.0021561, -0.0031076, 0.0028986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017089, upper bound: 0.0016403
time: 1.69 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017089, upper bound: 0.0016403
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0026973, 0.0005423, -0.0025997, 0.0006614, -0.0029241, 0.0026795
1: -0.0045433, -0.0035005, -0.0045261, -0.0034424, -0.0009998, 0.0009284
2: 0.0111866, 0.0155113, 0.0112994, 0.0156945, -0.0038365, 0.0034964
3: 1.0071212, 1.0098197, 1.0070002, 1.0097766, -0.0026554, 0.0028195
4: -0.0041556, -0.0034495, -0.0041897, -0.0034652, -0.0005572, 0.0006151
5: 0.0018906, 0.0043849, 0.0019641, 0.0044787, -0.0022459, 0.0020563
6: -0.0025812, -0.0023167, -0.0025822, -0.0023202, -0.0002610, 0.0002655
7: -0.0130595, -0.0086202, -0.0130734, -0.0088626, -0.0041484, 0.0044097
8: -0.0125813, -0.0049419, -0.0129783, -0.0050936, -0.0059413, 0.0065943
9: -0.0017610, 0.0020194, -0.0016966, 0.0022316, -0.0032314, 0.0028891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015719, upper bound: 0.0015923
time: 1.53 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016476, upper bound: 0.0015656
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0026848, 0.0006050, -0.0025488, 0.0005918, -0.0027390, 0.0026287
1: -0.0045497, -0.0034705, -0.0045161, -0.0034773, -0.0009277, 0.0009044
2: 0.0111892, 0.0156078, 0.0113600, 0.0155874, -0.0035682, 0.0034360
3: 1.0070637, 1.0098355, 1.0070835, 1.0097517, -0.0026879, 0.0027519
4: -0.0041735, -0.0034477, -0.0041697, -0.0034735, -0.0005487, 0.0005677
5: 0.0018989, 0.0044343, 0.0020027, 0.0044238, -0.0021016, 0.0020178
6: -0.0025806, -0.0023154, -0.0025761, -0.0023223, -0.0002583, 0.0002607
7: -0.0130668, -0.0087306, -0.0130653, -0.0089641, -0.0040470, 0.0042774
8: -0.0127904, -0.0049096, -0.0127462, -0.0051747, -0.0058621, 0.0060511
9: -0.0017836, 0.0021312, -0.0016595, 0.0021076, -0.0029414, 0.0028591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016571, upper bound: 0.0017234
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016432, upper bound: 0.0017234
time: 1.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028438, 0.0005509, -0.0025488, 0.0005918, -0.0029290, 0.0026211
1: -0.0045801, -0.0034958, -0.0045161, -0.0034773, -0.0009785, 0.0009034
2: 0.0110026, 0.0155245, 0.0113600, 0.0155874, -0.0038088, 0.0034242
3: 1.0071018, 1.0099113, 1.0070835, 1.0097517, -0.0026499, 0.0028278
4: -0.0041580, -0.0034214, -0.0041697, -0.0034735, -0.0005465, 0.0006054
5: 0.0017787, 0.0043916, 0.0020027, 0.0044238, -0.0022468, 0.0020118
6: -0.0025861, -0.0023091, -0.0025761, -0.0023223, -0.0002639, 0.0002670
7: -0.0130605, -0.0083745, -0.0130653, -0.0089641, -0.0040461, 0.0046374
8: -0.0126099, -0.0046524, -0.0127462, -0.0051747, -0.0058366, 0.0064446
9: -0.0018965, 0.0020347, -0.0016595, 0.0021076, -0.0031288, 0.0028454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016571, upper bound: 0.0017234
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016432, upper bound: 0.0017234
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0026506, 0.0006190, -0.0027114, 0.0005442, -0.0027010, 0.0028611
1: -0.0045406, -0.0034639, -0.0045473, -0.0034995, -0.0009314, 0.0009676
2: 0.0112330, 0.0156293, 0.0111683, 0.0155142, -0.0035204, 0.0037360
3: 1.0070486, 1.0098127, 1.0071180, 1.0098295, -0.0027809, 0.0026947
4: -0.0041775, -0.0034545, -0.0041561, -0.0034466, -0.0005958, 0.0005601
5: 0.0019250, 0.0044453, 0.0018798, 0.0043864, -0.0020724, 0.0021960
6: -0.0025801, -0.0023173, -0.0025817, -0.0023159, -0.0002643, 0.0002645
7: -0.0130685, -0.0087912, -0.0130597, -0.0085972, -0.0044238, 0.0042165
8: -0.0128369, -0.0049803, -0.0125876, -0.0049117, -0.0063651, 0.0059670
9: -0.0017500, 0.0021561, -0.0017754, 0.0020228, -0.0028986, 0.0031076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016403, upper bound: 0.0017088
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016403, upper bound: 0.0017088
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0025997, 0.0006614, -0.0026973, 0.0005423, -0.0026795, 0.0029241
1: -0.0045261, -0.0034424, -0.0045433, -0.0035005, -0.0009284, 0.0009998
2: 0.0112994, 0.0156945, 0.0111866, 0.0155113, -0.0034964, 0.0038365
3: 1.0070002, 1.0097766, 1.0071212, 1.0098197, -0.0028195, 0.0026554
4: -0.0041897, -0.0034652, -0.0041556, -0.0034495, -0.0006151, 0.0005572
5: 0.0019641, 0.0044787, 0.0018906, 0.0043849, -0.0020563, 0.0022459
6: -0.0025822, -0.0023202, -0.0025812, -0.0023167, -0.0002655, 0.0002610
7: -0.0130734, -0.0088626, -0.0130595, -0.0086202, -0.0044097, 0.0041484
8: -0.0129783, -0.0050936, -0.0125813, -0.0049419, -0.0065943, 0.0059413
9: -0.0016966, 0.0022316, -0.0017610, 0.0020194, -0.0028891, 0.0032314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015923, upper bound: 0.0016490
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016476
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0026848, 0.0006050, -0.0026848, 0.0006050, -0.0027325, 0.0027325
1: -0.0045497, -0.0034705, -0.0045497, -0.0034705, -0.0009430, 0.0009430
2: 0.0111892, 0.0156078, 0.0111892, 0.0156078, -0.0035604, 0.0035604
3: 1.0070637, 1.0098355, 1.0070637, 1.0098355, -0.0027717, 0.0027717
4: -0.0041735, -0.0034477, -0.0041735, -0.0034477, -0.0005670, 0.0005670
5: 0.0018989, 0.0044343, 0.0018989, 0.0044343, -0.0020967, 0.0020967
6: -0.0025806, -0.0023154, -0.0025806, -0.0023154, -0.0002652, 0.0002652
7: -0.0130668, -0.0087306, -0.0130668, -0.0087306, -0.0042768, 0.0042768
8: -0.0127904, -0.0049096, -0.0127904, -0.0049096, -0.0060479, 0.0060479
9: -0.0017836, 0.0021312, -0.0017836, 0.0021312, -0.0029467, 0.0029467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017203
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017176
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0026848, 0.0006050, -0.0028438, 0.0005509, -0.0027389, 0.0029319
1: -0.0045497, -0.0034705, -0.0045801, -0.0034958, -0.0009464, 0.0009970
2: 0.0111892, 0.0156078, 0.0110026, 0.0155245, -0.0035703, 0.0038144
3: 1.0070637, 1.0098355, 1.0071018, 1.0099113, -0.0028476, 0.0027337
4: -0.0041735, -0.0034477, -0.0041580, -0.0034214, -0.0006058, 0.0005688
5: 0.0018989, 0.0044343, 0.0017787, 0.0043916, -0.0021018, 0.0022491
6: -0.0025806, -0.0023154, -0.0025861, -0.0023091, -0.0002714, 0.0002708
7: -0.0130668, -0.0087306, -0.0130605, -0.0083745, -0.0046367, 0.0042776
8: -0.0127904, -0.0049096, -0.0126099, -0.0046524, -0.0064552, 0.0060693
9: -0.0017836, 0.0021312, -0.0018965, 0.0020347, -0.0029581, 0.0031452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017203
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017176
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028438, 0.0005509, -0.0026506, 0.0006190, -0.0029647, 0.0026905
1: -0.0045801, -0.0034958, -0.0045406, -0.0034639, -0.0010077, 0.0009446
2: 0.0110026, 0.0155245, 0.0112330, 0.0156293, -0.0038648, 0.0035036
3: 1.0071018, 1.0099113, 1.0070486, 1.0098127, -0.0027109, 0.0028627
4: -0.0041580, -0.0034214, -0.0041775, -0.0034545, -0.0005579, 0.0006152
5: 0.0017787, 0.0043916, 0.0019250, 0.0044453, -0.0022749, 0.0020642
6: -0.0025861, -0.0023091, -0.0025801, -0.0023173, -0.0002689, 0.0002710
7: -0.0130605, -0.0083745, -0.0130685, -0.0087912, -0.0042152, 0.0046406
8: -0.0126099, -0.0046524, -0.0128369, -0.0049803, -0.0059489, 0.0065644
9: -0.0018965, 0.0020347, -0.0017500, 0.0021561, -0.0032036, 0.0028967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016297, upper bound: 0.0016817
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016289, upper bound: 0.0016628
time: 1.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028300, 0.0005490, -0.0025997, 0.0006614, -0.0030223, 0.0026695
1: -0.0045764, -0.0034968, -0.0045261, -0.0034424, -0.0010382, 0.0009426
2: 0.0110200, 0.0155217, 0.0112994, 0.0156945, -0.0039566, 0.0034830
3: 1.0071049, 1.0099020, 1.0070002, 1.0097766, -0.0026717, 0.0029018
4: -0.0041575, -0.0034241, -0.0041897, -0.0034652, -0.0005555, 0.0006328
5: 0.0017892, 0.0043902, 0.0019641, 0.0044787, -0.0023206, 0.0020488
6: -0.0025856, -0.0023099, -0.0025822, -0.0023202, -0.0002654, 0.0002723
7: -0.0130602, -0.0083966, -0.0130734, -0.0088626, -0.0041469, 0.0046267
8: -0.0126038, -0.0046812, -0.0129783, -0.0050936, -0.0059304, 0.0067749
9: -0.0018828, 0.0020315, -0.0016966, 0.0022316, -0.0033174, 0.0028910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016195, upper bound: 0.0016816
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016186, upper bound: 0.0016623
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027076, 0.0006082, -0.0025373, 0.0006491, -0.0028803, 0.0026671
1: -0.0045560, -0.0034688, -0.0045111, -0.0034495, -0.0009751, 0.0009284
2: 0.0111600, 0.0156127, 0.0113766, 0.0156755, -0.0037798, 0.0034944
3: 1.0070580, 1.0098512, 1.0070192, 1.0097394, -0.0026814, 0.0028321
4: -0.0041745, -0.0034432, -0.0041862, -0.0034765, -0.0005602, 0.0006064
5: 0.0018815, 0.0044368, 0.0020116, 0.0044689, -0.0022123, 0.0020481
6: -0.0025817, -0.0023141, -0.0025788, -0.0023233, -0.0002584, 0.0002647
7: -0.0130672, -0.0086821, -0.0130720, -0.0089798, -0.0040382, 0.0043408
8: -0.0128011, -0.0048620, -0.0129372, -0.0052096, -0.0060018, 0.0064966
9: -0.0018069, 0.0021369, -0.0016415, 0.0022096, -0.0031836, 0.0029332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016795, upper bound: 0.0017033
time: 1.18 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016795, upper bound: 0.0017474
time: 1.51 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0026930, 0.0006064, -0.0024862, 0.0006808, -0.0029285, 0.0026436
1: -0.0045521, -0.0034698, -0.0044956, -0.0034338, -0.0010039, 0.0009261
2: 0.0111786, 0.0156099, 0.0114451, 0.0157242, -0.0038574, 0.0034702
3: 1.0070612, 1.0098413, 1.0069802, 1.0097008, -0.0026395, 0.0028611
4: -0.0041739, -0.0034461, -0.0041952, -0.0034878, -0.0005576, 0.0006214
5: 0.0018926, 0.0044353, 0.0020509, 0.0044939, -0.0022506, 0.0020306
6: -0.0025811, -0.0023149, -0.0025805, -0.0023265, -0.0002546, 0.0002656
7: -0.0130670, -0.0087066, -0.0130757, -0.0090560, -0.0039644, 0.0043234
8: -0.0127949, -0.0048925, -0.0130427, -0.0053284, -0.0059797, 0.0066767
9: -0.0017923, 0.0021336, -0.0015845, 0.0022660, -0.0032816, 0.0029263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A1_B2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016720, upper bound: 0.0017033
time: 1.31 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016720, upper bound: 0.0017473
time: 1.57 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028679, 0.0005541, -0.0025373, 0.0006491, -0.0030746, 0.0026587
1: -0.0045864, -0.0034941, -0.0045111, -0.0034495, -0.0010281, 0.0009269
2: 0.0109718, 0.0155295, 0.0113766, 0.0156755, -0.0040269, 0.0034814
3: 1.0070959, 1.0099272, 1.0070192, 1.0097394, -0.0026435, 0.0029080
4: -0.0041590, -0.0034167, -0.0041862, -0.0034765, -0.0005578, 0.0006447
5: 0.0017601, 0.0043942, 0.0020116, 0.0044689, -0.0023610, 0.0020415
6: -0.0025873, -0.0023078, -0.0025788, -0.0023233, -0.0002640, 0.0002710
7: -0.0130608, -0.0083257, -0.0130720, -0.0089798, -0.0040372, 0.0047025
8: -0.0126208, -0.0046046, -0.0129372, -0.0052096, -0.0059737, 0.0069072
9: -0.0019199, 0.0020406, -0.0016415, 0.0022096, -0.0033790, 0.0029182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015989, upper bound: 0.0017046
time: 1.35 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015980, upper bound: 0.0016929
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028540, 0.0005523, -0.0024862, 0.0006808, -0.0031228, 0.0026352
1: -0.0045826, -0.0034951, -0.0044956, -0.0034338, -0.0010568, 0.0009246
2: 0.0109897, 0.0155267, 0.0114451, 0.0157242, -0.0041046, 0.0034572
3: 1.0070990, 1.0099176, 1.0069802, 1.0097008, -0.0026017, 0.0029374
4: -0.0041584, -0.0034195, -0.0041952, -0.0034878, -0.0005551, 0.0006598
5: 0.0017708, 0.0043927, 0.0020509, 0.0044939, -0.0023992, 0.0020239
6: -0.0025868, -0.0023086, -0.0025805, -0.0023265, -0.0002603, 0.0002719
7: -0.0130606, -0.0083486, -0.0130757, -0.0090560, -0.0039634, 0.0046862
8: -0.0126146, -0.0046337, -0.0130427, -0.0053284, -0.0059515, 0.0070870
9: -0.0019060, 0.0020373, -0.0015845, 0.0022660, -0.0034770, 0.0029113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015793, upper bound: 0.0017044
time: 1.34 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015774, upper bound: 0.0016917
time: 1.51 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0025498, 0.0006176, -0.0027134, 0.0006075, -0.0027267, 0.0028837
1: -0.0045162, -0.0034652, -0.0045452, -0.0034689, -0.0009508, 0.0009806
2: 0.0113588, 0.0156271, 0.0111685, 0.0156116, -0.0035881, 0.0037728
3: 1.0070552, 1.0097520, 1.0070500, 1.0098244, -0.0027692, 0.0027020
4: -0.0041771, -0.0034734, -0.0041742, -0.0034472, -0.0006035, 0.0005773
5: 0.0020019, 0.0044442, 0.0018784, 0.0044362, -0.0020950, 0.0022139
6: -0.0025774, -0.0023223, -0.0025846, -0.0023163, -0.0002611, 0.0002624
7: -0.0130683, -0.0089622, -0.0130671, -0.0085906, -0.0044307, 0.0040606
8: -0.0128323, -0.0051732, -0.0127986, -0.0049214, -0.0064530, 0.0061987
9: -0.0016600, 0.0021536, -0.0017681, 0.0021356, -0.0030461, 0.0031503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016102, upper bound: 0.0016027
time: 1.28 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015878, upper bound: 0.0015988
time: 1.51 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0026857, 0.0006278, -0.0027134, 0.0006075, -0.0028620, 0.0029019
1: -0.0045498, -0.0034595, -0.0045452, -0.0034689, -0.0009864, 0.0009894
2: 0.0111880, 0.0156427, 0.0111685, 0.0156116, -0.0037594, 0.0038008
3: 1.0070359, 1.0098357, 1.0070500, 1.0098244, -0.0027885, 0.0027857
4: -0.0041800, -0.0034476, -0.0041742, -0.0034472, -0.0006087, 0.0006035
5: 0.0018981, 0.0044522, 0.0018784, 0.0044362, -0.0021985, 0.0022282
6: -0.0025818, -0.0023153, -0.0025846, -0.0023163, -0.0002655, 0.0002693
7: -0.0130695, -0.0087289, -0.0130671, -0.0085906, -0.0044328, 0.0042934
8: -0.0128662, -0.0049082, -0.0127986, -0.0049214, -0.0065136, 0.0064710
9: -0.0017841, 0.0021717, -0.0017681, 0.0021356, -0.0031770, 0.0031827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015926, upper bound: 0.0016962
time: 1.25 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015878, upper bound: 0.0016773
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0025351, 0.0006157, -0.0026691, 0.0006386, -0.0027733, 0.0028624
1: -0.0045123, -0.0034662, -0.0045308, -0.0034533, -0.0009804, 0.0009782
2: 0.0113774, 0.0156242, 0.0112277, 0.0156594, -0.0036637, 0.0037500
3: 1.0070584, 1.0097424, 1.0070105, 1.0097885, -0.0027301, 0.0027319
4: -0.0041766, -0.0034762, -0.0041832, -0.0034571, -0.0006009, 0.0005920
5: 0.0020131, 0.0044427, 0.0019126, 0.0044607, -0.0021321, 0.0021979
6: -0.0025769, -0.0023231, -0.0025863, -0.0023193, -0.0002576, 0.0002632
7: -0.0130681, -0.0089879, -0.0130708, -0.0086514, -0.0043711, 0.0040422
8: -0.0128259, -0.0052032, -0.0129023, -0.0050283, -0.0064307, 0.0063749
9: -0.0016458, 0.0021502, -0.0017153, 0.0021910, -0.0031418, 0.0031430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015923, upper bound: 0.0016015
time: 1.44 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0015956
time: 1.47 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0026712, 0.0006259, -0.0026691, 0.0006386, -0.0029091, 0.0028808
1: -0.0045460, -0.0034605, -0.0045308, -0.0034533, -0.0010162, 0.0009871
2: 0.0112065, 0.0156399, 0.0112277, 0.0156594, -0.0038354, 0.0037784
3: 1.0070388, 1.0098261, 1.0070105, 1.0097885, -0.0027497, 0.0028156
4: -0.0041795, -0.0034505, -0.0041832, -0.0034571, -0.0006061, 0.0006182
5: 0.0019093, 0.0044507, 0.0019126, 0.0044607, -0.0022358, 0.0022125
6: -0.0025813, -0.0023161, -0.0025863, -0.0023193, -0.0002620, 0.0002701
7: -0.0130693, -0.0087528, -0.0130708, -0.0086514, -0.0043733, 0.0042763
8: -0.0128601, -0.0049382, -0.0129023, -0.0050283, -0.0064921, 0.0066474
9: -0.0017699, 0.0021684, -0.0017153, 0.0021910, -0.0032732, 0.0031759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015719, upper bound: 0.0016943
time: 1.44 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016754
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0025488, 0.0005918, -0.0027117, 0.0006629, -0.0027387, 0.0027879
1: -0.0045161, -0.0034773, -0.0045552, -0.0034419, -0.0009470, 0.0009455
2: 0.0113600, 0.0155874, 0.0111577, 0.0156968, -0.0036068, 0.0036364
3: 1.0070835, 1.0097517, 1.0069911, 1.0098493, -0.0027658, 0.0027605
4: -0.0041697, -0.0034735, -0.0041901, -0.0034432, -0.0005791, 0.0005808
5: 0.0020027, 0.0044238, 0.0018785, 0.0044798, -0.0021045, 0.0021394
6: -0.0025761, -0.0023223, -0.0025845, -0.0023142, -0.0002619, 0.0002622
7: -0.0130653, -0.0089641, -0.0130736, -0.0086833, -0.0043269, 0.0040603
8: -0.0127462, -0.0051747, -0.0129833, -0.0048635, -0.0061762, 0.0062403
9: -0.0016595, 0.0021076, -0.0018043, 0.0022343, -0.0030689, 0.0030071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016815
time: 1.21 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016692
time: 1.73 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0025488, 0.0005918, -0.0029294, 0.0006169, -0.0027301, 0.0030391
1: -0.0045161, -0.0034773, -0.0045988, -0.0034638, -0.0009502, 0.0010134
2: 0.0113600, 0.0155874, 0.0108971, 0.0156260, -0.0035936, 0.0039561
3: 1.0070835, 1.0097517, 1.0070214, 1.0099580, -0.0028745, 0.0027303
4: -0.0041697, -0.0034735, -0.0041769, -0.0034060, -0.0006297, 0.0005784
5: 0.0020027, 0.0044238, 0.0017135, 0.0044436, -0.0020977, 0.0023314
6: -0.0025761, -0.0023223, -0.0025917, -0.0023053, -0.0002708, 0.0002694
7: -0.0130653, -0.0089641, -0.0130682, -0.0082207, -0.0047928, 0.0040593
8: -0.0127462, -0.0051747, -0.0128299, -0.0044966, -0.0067099, 0.0062117
9: -0.0016595, 0.0021076, -0.0019662, 0.0021523, -0.0030536, 0.0032578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016815
time: 1.45 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016692
time: 1.56 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027114, 0.0005442, -0.0026704, 0.0006790, -0.0029677, 0.0027459
1: -0.0045473, -0.0034995, -0.0045447, -0.0034345, -0.0010097, 0.0009489
2: 0.0111683, 0.0155142, 0.0112097, 0.0157215, -0.0039046, 0.0035820
3: 1.0071180, 1.0098295, 1.0069755, 1.0098230, -0.0027050, 0.0028540
4: -0.0041561, -0.0034466, -0.0041947, -0.0034512, -0.0005705, 0.0006278
5: 0.0018798, 0.0043864, 0.0019100, 0.0044925, -0.0022802, 0.0021072
6: -0.0025817, -0.0023159, -0.0025839, -0.0023164, -0.0002653, 0.0002680
7: -0.0130597, -0.0085972, -0.0130755, -0.0087559, -0.0042550, 0.0044365
8: -0.0125876, -0.0049117, -0.0130369, -0.0049460, -0.0060847, 0.0067448
9: -0.0017754, 0.0020228, -0.0017653, 0.0022629, -0.0033127, 0.0029628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016554, upper bound: 0.0016193
time: 1.26 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015881, upper bound: 0.0015984
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0026973, 0.0005423, -0.0026210, 0.0007167, -0.0029888, 0.0027216
1: -0.0045433, -0.0035005, -0.0045294, -0.0034151, -0.0010239, 0.0009457
2: 0.0111866, 0.0155113, 0.0112755, 0.0157794, -0.0039359, 0.0035559
3: 1.0071212, 1.0098197, 1.0069301, 1.0097851, -0.0026639, 0.0028896
4: -0.0041556, -0.0034495, -0.0042055, -0.0034618, -0.0005674, 0.0006336
5: 0.0018906, 0.0043849, 0.0019480, 0.0045221, -0.0022968, 0.0020891
6: -0.0025812, -0.0023167, -0.0025859, -0.0023195, -0.0002616, 0.0002692
7: -0.0130595, -0.0086202, -0.0130799, -0.0088269, -0.0041850, 0.0044173
8: -0.0125813, -0.0049419, -0.0131624, -0.0050612, -0.0060569, 0.0068098
9: -0.0017610, 0.0020194, -0.0017092, 0.0023300, -0.0033465, 0.0029527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016490, upper bound: 0.0016181
time: 1.16 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016476, upper bound: 0.0015955
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0026848, 0.0006050, -0.0027117, 0.0006629, -0.0028501, 0.0027958
1: -0.0045497, -0.0034705, -0.0045552, -0.0034419, -0.0009845, 0.0009685
2: 0.0111892, 0.0156078, 0.0111577, 0.0156968, -0.0037435, 0.0036505
3: 1.0070637, 1.0098355, 1.0069911, 1.0098493, -0.0027856, 0.0028443
4: -0.0041735, -0.0034477, -0.0041901, -0.0034432, -0.0005832, 0.0006014
5: 0.0018989, 0.0044343, 0.0018785, 0.0044798, -0.0021893, 0.0021458
6: -0.0025806, -0.0023154, -0.0025845, -0.0023142, -0.0002663, 0.0002691
7: -0.0130668, -0.0087306, -0.0130736, -0.0086833, -0.0043264, 0.0042907
8: -0.0127904, -0.0049096, -0.0129833, -0.0048635, -0.0062342, 0.0064545
9: -0.0017836, 0.0021312, -0.0018043, 0.0022343, -0.0031704, 0.0030405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016777, upper bound: 0.0017434
time: 1.22 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016707, upper bound: 0.0017434
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028438, 0.0005509, -0.0027117, 0.0006629, -0.0030475, 0.0028022
1: -0.0045801, -0.0034958, -0.0045552, -0.0034419, -0.0010372, 0.0009719
2: 0.0110026, 0.0155245, 0.0111577, 0.0156968, -0.0039955, 0.0036603
3: 1.0071018, 1.0099113, 1.0069911, 1.0098493, -0.0027475, 0.0029202
4: -0.0041580, -0.0034214, -0.0041901, -0.0034432, -0.0005850, 0.0006402
5: 0.0017787, 0.0043916, 0.0018785, 0.0044798, -0.0023404, 0.0021509
6: -0.0025861, -0.0023091, -0.0025845, -0.0023142, -0.0002719, 0.0002754
7: -0.0130605, -0.0083745, -0.0130736, -0.0086833, -0.0043272, 0.0046501
8: -0.0126099, -0.0046524, -0.0129833, -0.0048635, -0.0062556, 0.0068632
9: -0.0018965, 0.0020347, -0.0018043, 0.0022343, -0.0033644, 0.0030520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016777, upper bound: 0.0017434
time: 1.51 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016707, upper bound: 0.0017434
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0026506, 0.0006190, -0.0029294, 0.0006169, -0.0028111, 0.0030876
1: -0.0045406, -0.0034639, -0.0045988, -0.0034638, -0.0009888, 0.0010519
2: 0.0112330, 0.0156293, 0.0108971, 0.0156260, -0.0036913, 0.0040343
3: 1.0070486, 1.0098127, 1.0070214, 1.0099580, -0.0029094, 0.0027913
4: -0.0041775, -0.0034545, -0.0041769, -0.0034060, -0.0006446, 0.0005932
5: 0.0019250, 0.0044453, 0.0017135, 0.0044436, -0.0021593, 0.0023700
6: -0.0025801, -0.0023173, -0.0025917, -0.0023053, -0.0002749, 0.0002744
7: -0.0130685, -0.0087912, -0.0130682, -0.0082207, -0.0047967, 0.0042294
8: -0.0128369, -0.0049803, -0.0128299, -0.0044966, -0.0068931, 0.0063652
9: -0.0017500, 0.0021561, -0.0019662, 0.0021523, -0.0031254, 0.0033662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016384, upper bound: 0.0016950
time: 1.51 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016191, upper bound: 0.0016945
time: 1.51 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0025997, 0.0006614, -0.0029139, 0.0006144, -0.0027865, 0.0031419
1: -0.0045261, -0.0034424, -0.0045946, -0.0034650, -0.0009854, 0.0010814
2: 0.0112994, 0.0156945, 0.0109170, 0.0156222, -0.0036656, 0.0041218
3: 1.0070002, 1.0097766, 1.0070252, 1.0099473, -0.0029471, 0.0027514
4: -0.0041897, -0.0034652, -0.0041762, -0.0034091, -0.0006616, 0.0005899
5: 0.0019641, 0.0044787, 0.0017253, 0.0044417, -0.0021410, 0.0024131
6: -0.0025822, -0.0023202, -0.0025910, -0.0023061, -0.0002761, 0.0002708
7: -0.0130734, -0.0088626, -0.0130679, -0.0082472, -0.0047782, 0.0041605
8: -0.0129783, -0.0050936, -0.0128218, -0.0045296, -0.0070943, 0.0063351
9: -0.0016966, 0.0022316, -0.0019505, 0.0021479, -0.0031148, 0.0034759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016384, upper bound: 0.0016908
time: 1.50 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016901
time: 1.55 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0025373, 0.0006491, -0.0027076, 0.0006082, -0.0026671, 0.0028803
1: -0.0045111, -0.0034495, -0.0045560, -0.0034688, -0.0009284, 0.0009751
2: 0.0113766, 0.0156755, 0.0111600, 0.0156127, -0.0034944, 0.0037798
3: 1.0070192, 1.0097394, 1.0070580, 1.0098512, -0.0028321, 0.0026814
4: -0.0041862, -0.0034765, -0.0041745, -0.0034432, -0.0006064, 0.0005602
5: 0.0020116, 0.0044689, 0.0018815, 0.0044368, -0.0020481, 0.0022123
6: -0.0025788, -0.0023233, -0.0025817, -0.0023141, -0.0002647, 0.0002584
7: -0.0130720, -0.0089798, -0.0130672, -0.0086821, -0.0043408, 0.0040382
8: -0.0129372, -0.0052096, -0.0128011, -0.0048620, -0.0064966, 0.0060018
9: -0.0016415, 0.0022096, -0.0018069, 0.0021369, -0.0029332, 0.0031836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016795
time: 1.40 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016795
time: 1.44 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0024862, 0.0006808, -0.0026930, 0.0006064, -0.0026436, 0.0029285
1: -0.0044956, -0.0034338, -0.0045521, -0.0034698, -0.0009261, 0.0010039
2: 0.0114451, 0.0157242, 0.0111786, 0.0156099, -0.0034702, 0.0038574
3: 1.0069802, 1.0097008, 1.0070612, 1.0098413, -0.0028611, 0.0026395
4: -0.0041952, -0.0034878, -0.0041739, -0.0034461, -0.0006214, 0.0005576
5: 0.0020509, 0.0044939, 0.0018926, 0.0044353, -0.0020306, 0.0022506
6: -0.0025805, -0.0023265, -0.0025811, -0.0023149, -0.0002656, 0.0002546
7: -0.0130757, -0.0090560, -0.0130670, -0.0087066, -0.0043234, 0.0039644
8: -0.0130427, -0.0053284, -0.0127949, -0.0048925, -0.0066767, 0.0059797
9: -0.0015845, 0.0022660, -0.0017923, 0.0021336, -0.0029263, 0.0032816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016720
time: 1.38 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016720
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0025373, 0.0006491, -0.0028679, 0.0005541, -0.0026587, 0.0030746
1: -0.0045111, -0.0034495, -0.0045864, -0.0034941, -0.0009269, 0.0010281
2: 0.0113766, 0.0156755, 0.0109718, 0.0155295, -0.0034814, 0.0040269
3: 1.0070192, 1.0097394, 1.0070959, 1.0099272, -0.0029080, 0.0026435
4: -0.0041862, -0.0034765, -0.0041590, -0.0034167, -0.0006447, 0.0005578
5: 0.0020116, 0.0044689, 0.0017601, 0.0043942, -0.0020415, 0.0023610
6: -0.0025788, -0.0023233, -0.0025873, -0.0023078, -0.0002710, 0.0002640
7: -0.0130720, -0.0089798, -0.0130608, -0.0083257, -0.0047025, 0.0040372
8: -0.0129372, -0.0052096, -0.0126208, -0.0046046, -0.0069072, 0.0059737
9: -0.0016415, 0.0022096, -0.0019199, 0.0020406, -0.0029182, 0.0033790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017046, upper bound: 0.0015989
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016929, upper bound: 0.0015980
time: 1.55 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0024862, 0.0006808, -0.0028540, 0.0005523, -0.0026352, 0.0031228
1: -0.0044956, -0.0034338, -0.0045826, -0.0034951, -0.0009246, 0.0010568
2: 0.0114451, 0.0157242, 0.0109897, 0.0155267, -0.0034572, 0.0041046
3: 1.0069802, 1.0097008, 1.0070990, 1.0099176, -0.0029374, 0.0026017
4: -0.0041952, -0.0034878, -0.0041584, -0.0034195, -0.0006598, 0.0005551
5: 0.0020509, 0.0044939, 0.0017708, 0.0043927, -0.0020239, 0.0023992
6: -0.0025805, -0.0023265, -0.0025868, -0.0023086, -0.0002719, 0.0002603
7: -0.0130757, -0.0090560, -0.0130606, -0.0083486, -0.0046862, 0.0039634
8: -0.0130427, -0.0053284, -0.0126146, -0.0046337, -0.0070870, 0.0059515
9: -0.0015845, 0.0022660, -0.0019060, 0.0020373, -0.0029113, 0.0034771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017044, upper bound: 0.0015793
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016918, upper bound: 0.0015774
time: 1.40 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027134, 0.0006075, -0.0025498, 0.0006176, -0.0028837, 0.0027267
1: -0.0045452, -0.0034689, -0.0045162, -0.0034652, -0.0009806, 0.0009508
2: 0.0111685, 0.0156116, 0.0113588, 0.0156271, -0.0037728, 0.0035881
3: 1.0070500, 1.0098244, 1.0070552, 1.0097520, -0.0027020, 0.0027692
4: -0.0041742, -0.0034472, -0.0041771, -0.0034734, -0.0005773, 0.0006035
5: 0.0018784, 0.0044362, 0.0020019, 0.0044442, -0.0022139, 0.0020950
6: -0.0025846, -0.0023163, -0.0025774, -0.0023223, -0.0002624, 0.0002611
7: -0.0130671, -0.0085906, -0.0130683, -0.0089622, -0.0040606, 0.0044307
8: -0.0127986, -0.0049214, -0.0128323, -0.0051732, -0.0061987, 0.0064530
9: -0.0017681, 0.0021356, -0.0016600, 0.0021536, -0.0031503, 0.0030461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016027, upper bound: 0.0016102
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015988, upper bound: 0.0015878
time: 1.46 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027134, 0.0006075, -0.0026857, 0.0006278, -0.0029019, 0.0028620
1: -0.0045452, -0.0034689, -0.0045498, -0.0034595, -0.0009894, 0.0009864
2: 0.0111685, 0.0156116, 0.0111880, 0.0156427, -0.0038008, 0.0037594
3: 1.0070500, 1.0098244, 1.0070359, 1.0098357, -0.0027857, 0.0027885
4: -0.0041742, -0.0034472, -0.0041800, -0.0034476, -0.0006035, 0.0006087
5: 0.0018784, 0.0044362, 0.0018981, 0.0044522, -0.0022282, 0.0021985
6: -0.0025846, -0.0023163, -0.0025818, -0.0023153, -0.0002693, 0.0002655
7: -0.0130671, -0.0085906, -0.0130695, -0.0087289, -0.0042934, 0.0044328
8: -0.0127986, -0.0049214, -0.0128662, -0.0049082, -0.0064710, 0.0065136
9: -0.0017681, 0.0021356, -0.0017841, 0.0021717, -0.0031827, 0.0031770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016212, upper bound: 0.0015926
time: 1.63 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015988, upper bound: 0.0015878
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0026691, 0.0006386, -0.0025351, 0.0006157, -0.0028624, 0.0027733
1: -0.0045308, -0.0034533, -0.0045123, -0.0034662, -0.0009782, 0.0009804
2: 0.0112277, 0.0156594, 0.0113774, 0.0156242, -0.0037500, 0.0036637
3: 1.0070105, 1.0097885, 1.0070584, 1.0097424, -0.0027319, 0.0027301
4: -0.0041832, -0.0034571, -0.0041766, -0.0034762, -0.0005920, 0.0006009
5: 0.0019126, 0.0044607, 0.0020131, 0.0044427, -0.0021979, 0.0021321
6: -0.0025863, -0.0023193, -0.0025769, -0.0023231, -0.0002632, 0.0002576
7: -0.0130708, -0.0086514, -0.0130681, -0.0089879, -0.0040422, 0.0043711
8: -0.0129023, -0.0050283, -0.0128259, -0.0052032, -0.0063749, 0.0064307
9: -0.0017153, 0.0021910, -0.0016458, 0.0021502, -0.0031430, 0.0031418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016015, upper bound: 0.0015923
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015955, upper bound: 0.0015656
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0026691, 0.0006386, -0.0026712, 0.0006259, -0.0028808, 0.0029091
1: -0.0045308, -0.0034533, -0.0045460, -0.0034605, -0.0009871, 0.0010162
2: 0.0112277, 0.0156594, 0.0112065, 0.0156399, -0.0037784, 0.0038354
3: 1.0070105, 1.0097885, 1.0070388, 1.0098261, -0.0028156, 0.0027497
4: -0.0041832, -0.0034571, -0.0041795, -0.0034505, -0.0006182, 0.0006061
5: 0.0019126, 0.0044607, 0.0019093, 0.0044507, -0.0022125, 0.0022358
6: -0.0025863, -0.0023193, -0.0025813, -0.0023161, -0.0002701, 0.0002620
7: -0.0130708, -0.0086514, -0.0130693, -0.0087528, -0.0042763, 0.0043733
8: -0.0129023, -0.0050283, -0.0128601, -0.0049382, -0.0066474, 0.0064921
9: -0.0017153, 0.0021910, -0.0017699, 0.0021684, -0.0031759, 0.0032732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016194, upper bound: 0.0015719
time: 1.62 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015955, upper bound: 0.0015656
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0027117, 0.0006629, -0.0025488, 0.0005918, -0.0027879, 0.0027387
1: -0.0045552, -0.0034419, -0.0045161, -0.0034773, -0.0009455, 0.0009470
2: 0.0111577, 0.0156968, 0.0113600, 0.0155874, -0.0036364, 0.0036068
3: 1.0069911, 1.0098493, 1.0070835, 1.0097517, -0.0027605, 0.0027658
4: -0.0041901, -0.0034432, -0.0041697, -0.0034735, -0.0005808, 0.0005791
5: 0.0018785, 0.0044798, 0.0020027, 0.0044238, -0.0021394, 0.0021045
6: -0.0025845, -0.0023142, -0.0025761, -0.0023223, -0.0002622, 0.0002619
7: -0.0130736, -0.0086833, -0.0130653, -0.0089641, -0.0040603, 0.0043269
8: -0.0129833, -0.0048635, -0.0127462, -0.0051747, -0.0062403, 0.0061762
9: -0.0018043, 0.0022343, -0.0016595, 0.0021076, -0.0030071, 0.0030689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016815, upper bound: 0.0017232
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016692, upper bound: 0.0017232
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0029294, 0.0006169, -0.0025488, 0.0005918, -0.0030391, 0.0027301
1: -0.0045988, -0.0034638, -0.0045161, -0.0034773, -0.0010134, 0.0009502
2: 0.0108971, 0.0156260, 0.0113600, 0.0155874, -0.0039561, 0.0035936
3: 1.0070214, 1.0099580, 1.0070835, 1.0097517, -0.0027303, 0.0028745
4: -0.0041769, -0.0034060, -0.0041697, -0.0034735, -0.0005784, 0.0006297
5: 0.0017135, 0.0044436, 0.0020027, 0.0044238, -0.0023314, 0.0020977
6: -0.0025917, -0.0023053, -0.0025761, -0.0023223, -0.0002694, 0.0002708
7: -0.0130682, -0.0082207, -0.0130653, -0.0089641, -0.0040593, 0.0047928
8: -0.0128299, -0.0044966, -0.0127462, -0.0051747, -0.0062117, 0.0067099
9: -0.0019662, 0.0021523, -0.0016595, 0.0021076, -0.0032578, 0.0030536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016815, upper bound: 0.0017232
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016692, upper bound: 0.0017232
time: 1.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0026704, 0.0006790, -0.0027114, 0.0005442, -0.0027459, 0.0029677
1: -0.0045447, -0.0034345, -0.0045473, -0.0034995, -0.0009489, 0.0010097
2: 0.0112097, 0.0157215, 0.0111683, 0.0155142, -0.0035820, 0.0039046
3: 1.0069755, 1.0098230, 1.0071180, 1.0098295, -0.0028540, 0.0027050
4: -0.0041947, -0.0034512, -0.0041561, -0.0034466, -0.0006278, 0.0005705
5: 0.0019100, 0.0044925, 0.0018798, 0.0043864, -0.0021072, 0.0022802
6: -0.0025839, -0.0023164, -0.0025817, -0.0023159, -0.0002680, 0.0002653
7: -0.0130755, -0.0087559, -0.0130597, -0.0085972, -0.0044365, 0.0042550
8: -0.0130369, -0.0049460, -0.0125876, -0.0049117, -0.0067448, 0.0060847
9: -0.0017653, 0.0022629, -0.0017754, 0.0020228, -0.0029628, 0.0033127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016193, upper bound: 0.0016554
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015984, upper bound: 0.0016542
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0026210, 0.0007167, -0.0026973, 0.0005423, -0.0027216, 0.0029888
1: -0.0045294, -0.0034151, -0.0045433, -0.0035005, -0.0009457, 0.0010239
2: 0.0112755, 0.0157794, 0.0111866, 0.0155113, -0.0035559, 0.0039359
3: 1.0069301, 1.0097851, 1.0071212, 1.0098197, -0.0028896, 0.0026639
4: -0.0042055, -0.0034618, -0.0041556, -0.0034495, -0.0006336, 0.0005674
5: 0.0019480, 0.0045221, 0.0018906, 0.0043849, -0.0020891, 0.0022968
6: -0.0025859, -0.0023195, -0.0025812, -0.0023167, -0.0002692, 0.0002616
7: -0.0130799, -0.0088269, -0.0130595, -0.0086202, -0.0044173, 0.0041850
8: -0.0131624, -0.0050612, -0.0125813, -0.0049419, -0.0068098, 0.0060569
9: -0.0017092, 0.0023300, -0.0017610, 0.0020194, -0.0029527, 0.0033465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016181, upper bound: 0.0016490
time: 1.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015956, upper bound: 0.0016476
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027117, 0.0006629, -0.0026848, 0.0006050, -0.0027958, 0.0028501
1: -0.0045552, -0.0034419, -0.0045497, -0.0034705, -0.0009685, 0.0009845
2: 0.0111577, 0.0156968, 0.0111892, 0.0156078, -0.0036505, 0.0037435
3: 1.0069911, 1.0098493, 1.0070637, 1.0098355, -0.0028443, 0.0027856
4: -0.0041901, -0.0034432, -0.0041735, -0.0034477, -0.0006014, 0.0005832
5: 0.0018785, 0.0044798, 0.0018989, 0.0044343, -0.0021458, 0.0021893
6: -0.0025845, -0.0023142, -0.0025806, -0.0023154, -0.0002691, 0.0002663
7: -0.0130736, -0.0086833, -0.0130668, -0.0087306, -0.0042907, 0.0043264
8: -0.0129833, -0.0048635, -0.0127904, -0.0049096, -0.0064545, 0.0062342
9: -0.0018043, 0.0022343, -0.0017836, 0.0021312, -0.0030405, 0.0031704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017190
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017164
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027117, 0.0006629, -0.0028438, 0.0005509, -0.0028022, 0.0030475
1: -0.0045552, -0.0034419, -0.0045801, -0.0034958, -0.0009719, 0.0010372
2: 0.0111577, 0.0156968, 0.0110026, 0.0155245, -0.0036603, 0.0039955
3: 1.0069911, 1.0098493, 1.0071018, 1.0099113, -0.0029202, 0.0027475
4: -0.0041901, -0.0034432, -0.0041580, -0.0034214, -0.0006402, 0.0005850
5: 0.0018785, 0.0044798, 0.0017787, 0.0043916, -0.0021509, 0.0023404
6: -0.0025845, -0.0023142, -0.0025861, -0.0023091, -0.0002754, 0.0002719
7: -0.0130736, -0.0086833, -0.0130605, -0.0083745, -0.0046501, 0.0043272
8: -0.0129833, -0.0048635, -0.0126099, -0.0046524, -0.0068632, 0.0062556
9: -0.0018043, 0.0022343, -0.0018965, 0.0020347, -0.0030520, 0.0033644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017190
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017164
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0029294, 0.0006169, -0.0026506, 0.0006190, -0.0030876, 0.0028111
1: -0.0045988, -0.0034638, -0.0045406, -0.0034639, -0.0010519, 0.0009888
2: 0.0108971, 0.0156260, 0.0112330, 0.0156293, -0.0040343, 0.0036913
3: 1.0070214, 1.0099580, 1.0070486, 1.0098127, -0.0027913, 0.0029094
4: -0.0041769, -0.0034060, -0.0041775, -0.0034545, -0.0005932, 0.0006446
5: 0.0017135, 0.0044436, 0.0019250, 0.0044453, -0.0023700, 0.0021593
6: -0.0025917, -0.0023053, -0.0025801, -0.0023173, -0.0002744, 0.0002749
7: -0.0130682, -0.0082207, -0.0130685, -0.0087912, -0.0042294, 0.0047967
8: -0.0128299, -0.0044966, -0.0128369, -0.0049803, -0.0063652, 0.0068931
9: -0.0019662, 0.0021523, -0.0017500, 0.0021561, -0.0033662, 0.0031254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016601, upper bound: 0.0016816
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016601, upper bound: 0.0016628
time: 1.50 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0029139, 0.0006144, -0.0025997, 0.0006614, -0.0031419, 0.0027865
1: -0.0045946, -0.0034650, -0.0045261, -0.0034424, -0.0010814, 0.0009854
2: 0.0109170, 0.0156222, 0.0112994, 0.0156945, -0.0041218, 0.0036656
3: 1.0070252, 1.0099473, 1.0070002, 1.0097766, -0.0027514, 0.0029471
4: -0.0041762, -0.0034091, -0.0041897, -0.0034652, -0.0005899, 0.0006616
5: 0.0017253, 0.0044417, 0.0019641, 0.0044787, -0.0024131, 0.0021410
6: -0.0025910, -0.0023061, -0.0025822, -0.0023202, -0.0002708, 0.0002761
7: -0.0130679, -0.0082472, -0.0130734, -0.0088626, -0.0041605, 0.0047782
8: -0.0128218, -0.0045296, -0.0129783, -0.0050936, -0.0063351, 0.0070943
9: -0.0019505, 0.0021479, -0.0016966, 0.0022316, -0.0034759, 0.0031148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016515, upper bound: 0.0016815
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016514, upper bound: 0.0016623
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0025373, 0.0006491, -0.0027377, 0.0006664, -0.0027008, 0.0028463
1: -0.0045111, -0.0034495, -0.0045617, -0.0034401, -0.0009402, 0.0009633
2: 0.0113766, 0.0156755, 0.0111253, 0.0157021, -0.0035457, 0.0037164
3: 1.0070192, 1.0097394, 1.0069853, 1.0098656, -0.0028465, 0.0027541
4: -0.0041862, -0.0034765, -0.0041911, -0.0034384, -0.0005929, 0.0005694
5: 0.0020116, 0.0044689, 0.0018587, 0.0044825, -0.0020745, 0.0021846
6: -0.0025788, -0.0023233, -0.0025857, -0.0023129, -0.0002659, 0.0002624
7: -0.0130720, -0.0089798, -0.0130740, -0.0086327, -0.0043842, 0.0040440
8: -0.0129372, -0.0052096, -0.0129947, -0.0048128, -0.0063325, 0.0061051
9: -0.0016415, 0.0022096, -0.0018282, 0.0022404, -0.0029917, 0.0030897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016837
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016837
time: 1.73 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0024862, 0.0006808, -0.0027213, 0.0006639, -0.0026735, 0.0029040
1: -0.0044956, -0.0034338, -0.0045573, -0.0034413, -0.0009360, 0.0009924
2: 0.0114451, 0.0157242, 0.0111463, 0.0156983, -0.0035164, 0.0038096
3: 1.0069802, 1.0097008, 1.0069892, 1.0098544, -0.0028743, 0.0027115
4: -0.0041952, -0.0034878, -0.0041904, -0.0034417, -0.0006109, 0.0005656
5: 0.0020509, 0.0044939, 0.0018712, 0.0044806, -0.0020540, 0.0022304
6: -0.0025805, -0.0023265, -0.0025850, -0.0023138, -0.0002667, 0.0002585
7: -0.0130757, -0.0090560, -0.0130737, -0.0086603, -0.0043648, 0.0039694
8: -0.0130427, -0.0053284, -0.0129866, -0.0048474, -0.0065461, 0.0060699
9: -0.0015845, 0.0022660, -0.0018117, 0.0022361, -0.0029785, 0.0032065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016765
time: 1.43 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016765
time: 1.60 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0025373, 0.0006491, -0.0029551, 0.0006203, -0.0026869, 0.0030921
1: -0.0045111, -0.0034495, -0.0046055, -0.0034619, -0.0009423, 0.0010295
2: 0.0113766, 0.0156755, 0.0108646, 0.0156313, -0.0035243, 0.0040258
3: 1.0070192, 1.0097394, 1.0070151, 1.0099747, -0.0029556, 0.0027243
4: -0.0041862, -0.0034765, -0.0041779, -0.0034009, -0.0006416, 0.0005654
5: 0.0020116, 0.0044689, 0.0016938, 0.0044463, -0.0020636, 0.0023723
6: -0.0025788, -0.0023233, -0.0025929, -0.0023039, -0.0002749, 0.0002696
7: -0.0130720, -0.0089798, -0.0130686, -0.0081669, -0.0048520, 0.0040423
8: -0.0129372, -0.0052096, -0.0128413, -0.0044440, -0.0068460, 0.0060586
9: -0.0016415, 0.0022096, -0.0019910, 0.0021584, -0.0029669, 0.0033334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017099, upper bound: 0.0016109
time: 1.45 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017000, upper bound: 0.0016107
time: 1.68 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0024862, 0.0006808, -0.0029393, 0.0006179, -0.0026595, 0.0031498
1: -0.0044956, -0.0034338, -0.0046013, -0.0034632, -0.0009381, 0.0010584
2: 0.0114451, 0.0157242, 0.0108847, 0.0156275, -0.0034949, 0.0041187
3: 1.0069802, 1.0097008, 1.0070190, 1.0099641, -0.0029839, 0.0026817
4: -0.0041952, -0.0034878, -0.0041772, -0.0034041, -0.0006595, 0.0005616
5: 0.0020509, 0.0044939, 0.0017060, 0.0044444, -0.0020430, 0.0024181
6: -0.0025805, -0.0023265, -0.0025923, -0.0023048, -0.0002757, 0.0002658
7: -0.0130757, -0.0090560, -0.0130683, -0.0081935, -0.0048338, 0.0039678
8: -0.0130427, -0.0053284, -0.0128332, -0.0044771, -0.0070586, 0.0060233
9: -0.0015845, 0.0022660, -0.0019754, 0.0021540, -0.0029537, 0.0034495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017097, upper bound: 0.0015920
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016780, upper bound: 0.0015917
time: 1.78 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0027515, 0.0006175, -0.0025384, 0.0006783, -0.0029360, 0.0026654
1: -0.0045544, -0.0034639, -0.0045112, -0.0034360, -0.0009925, 0.0009372
2: 0.0111212, 0.0156270, 0.0113754, 0.0157204, -0.0038415, 0.0034909
3: 1.0070354, 1.0098472, 1.0069883, 1.0097396, -0.0027043, 0.0028589
4: -0.0041771, -0.0034401, -0.0041945, -0.0034764, -0.0005591, 0.0006142
5: 0.0018494, 0.0044441, 0.0020108, 0.0044919, -0.0022540, 0.0020466
6: -0.0025865, -0.0023144, -0.0025803, -0.0023233, -0.0002632, 0.0002659
7: -0.0130683, -0.0085250, -0.0130754, -0.0089779, -0.0040416, 0.0044987
8: -0.0128321, -0.0048495, -0.0130345, -0.0052081, -0.0059854, 0.0065737
9: -0.0018021, 0.0021535, -0.0016420, 0.0022617, -0.0032118, 0.0029274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016221, upper bound: 0.0016059
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016200, upper bound: 0.0015842
time: 1.46 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0027380, 0.0006151, -0.0024873, 0.0007091, -0.0029891, 0.0026376
1: -0.0045506, -0.0034651, -0.0044957, -0.0034203, -0.0010215, 0.0009328
2: 0.0111385, 0.0156233, 0.0114439, 0.0157677, -0.0039267, 0.0034609
3: 1.0070392, 1.0098377, 1.0069497, 1.0097010, -0.0026618, 0.0028881
4: -0.0041764, -0.0034428, -0.0042033, -0.0034876, -0.0005553, 0.0006306
5: 0.0018597, 0.0044422, 0.0020501, 0.0045161, -0.0022961, 0.0020257
6: -0.0025859, -0.0023152, -0.0025821, -0.0023265, -0.0002594, 0.0002670
7: -0.0130680, -0.0085470, -0.0130790, -0.0090538, -0.0039673, 0.0044844
8: -0.0128239, -0.0048782, -0.0131370, -0.0053268, -0.0059487, 0.0067689
9: -0.0017881, 0.0021491, -0.0015850, 0.0023164, -0.0033182, 0.0029133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016028, upper bound: 0.0016047
time: 1.50 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015990, upper bound: 0.0015819
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027134, 0.0006075, -0.0027136, 0.0006891, -0.0029227, 0.0028624
1: -0.0045452, -0.0034689, -0.0045556, -0.0034295, -0.0009953, 0.0009721
2: 0.0111685, 0.0156116, 0.0111553, 0.0157371, -0.0038309, 0.0037494
3: 1.0070500, 1.0098244, 1.0069607, 1.0098501, -0.0028001, 0.0028636
4: -0.0041742, -0.0034472, -0.0041976, -0.0034428, -0.0006007, 0.0006137
5: 0.0018784, 0.0044362, 0.0018770, 0.0045005, -0.0022445, 0.0021979
6: -0.0025846, -0.0023163, -0.0025859, -0.0023142, -0.0002705, 0.0002696
7: -0.0130671, -0.0085906, -0.0130767, -0.0086793, -0.0043395, 0.0044368
8: -0.0127986, -0.0049214, -0.0130706, -0.0048600, -0.0064337, 0.0065759
9: -0.0017681, 0.0021356, -0.0018057, 0.0022809, -0.0032173, 0.0031431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016855, upper bound: 0.0016226
time: 1.52 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016853, upper bound: 0.0016052
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0026691, 0.0006386, -0.0026967, 0.0006867, -0.0028979, 0.0028835
1: -0.0045308, -0.0034533, -0.0045510, -0.0034307, -0.0009906, 0.0010071
2: 0.0112277, 0.0156594, 0.0111773, 0.0157333, -0.0038026, 0.0037853
3: 1.0070105, 1.0097885, 1.0069647, 1.0098388, -0.0028284, 0.0028238
4: -0.0041832, -0.0034571, -0.0041969, -0.0034463, -0.0006076, 0.0006099
5: 0.0019126, 0.0044607, 0.0018900, 0.0044986, -0.0022257, 0.0022148
6: -0.0025863, -0.0023193, -0.0025852, -0.0023151, -0.0002711, 0.0002659
7: -0.0130708, -0.0086514, -0.0130764, -0.0087089, -0.0043152, 0.0043766
8: -0.0129023, -0.0050283, -0.0130625, -0.0048961, -0.0065120, 0.0065408
9: -0.0017153, 0.0021910, -0.0017885, 0.0022766, -0.0032029, 0.0031981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016999, upper bound: 0.0015852
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015990, upper bound: 0.0015819
time: 1.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0027117, 0.0006629, -0.0025728, 0.0006592, -0.0028276, 0.0027187
1: -0.0045552, -0.0034419, -0.0045204, -0.0034445, -0.0009638, 0.0009377
2: 0.0111577, 0.0156968, 0.0113323, 0.0156910, -0.0036956, 0.0035655
3: 1.0069911, 1.0098493, 1.0070047, 1.0097625, -0.0027714, 0.0028446
4: -0.0041901, -0.0034432, -0.0041890, -0.0034697, -0.0005718, 0.0005903
5: 0.0018785, 0.0044798, 0.0019845, 0.0044769, -0.0021705, 0.0020878
6: -0.0025845, -0.0023142, -0.0025805, -0.0023214, -0.0002631, 0.0002663
7: -0.0130736, -0.0086833, -0.0130732, -0.0089197, -0.0041020, 0.0043329
8: -0.0129833, -0.0048635, -0.0129709, -0.0051377, -0.0061243, 0.0063072
9: -0.0018043, 0.0022343, -0.0016757, 0.0022276, -0.0030857, 0.0029989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016820, upper bound: 0.0017307
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016695, upper bound: 0.0017307
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0029294, 0.0006169, -0.0025728, 0.0006592, -0.0030688, 0.0027058
1: -0.0045988, -0.0034638, -0.0045204, -0.0034445, -0.0010264, 0.0009405
2: 0.0108971, 0.0156260, 0.0113323, 0.0156910, -0.0039986, 0.0035457
3: 1.0070214, 1.0099580, 1.0070047, 1.0097625, -0.0027411, 0.0029533
4: -0.0041769, -0.0034060, -0.0041890, -0.0034697, -0.0005681, 0.0006375
5: 0.0017135, 0.0044436, 0.0019845, 0.0044769, -0.0023545, 0.0020777
6: -0.0025917, -0.0023053, -0.0025805, -0.0023214, -0.0002703, 0.0002753
7: -0.0130682, -0.0082207, -0.0130732, -0.0089197, -0.0041005, 0.0047977
8: -0.0128299, -0.0044966, -0.0129709, -0.0051377, -0.0060813, 0.0068063
9: -0.0019662, 0.0021523, -0.0016757, 0.0022276, -0.0033170, 0.0029759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016820, upper bound: 0.0017307
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016695, upper bound: 0.0017307
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0027136, 0.0006891, -0.0027134, 0.0006075, -0.0028624, 0.0029227
1: -0.0045556, -0.0034295, -0.0045452, -0.0034689, -0.0009721, 0.0009953
2: 0.0111553, 0.0157371, 0.0111685, 0.0156116, -0.0037494, 0.0038309
3: 1.0069607, 1.0098501, 1.0070500, 1.0098244, -0.0028636, 0.0028001
4: -0.0041976, -0.0034428, -0.0041742, -0.0034472, -0.0006137, 0.0006007
5: 0.0018770, 0.0045005, 0.0018784, 0.0044362, -0.0021979, 0.0022445
6: -0.0025859, -0.0023142, -0.0025846, -0.0023163, -0.0002696, 0.0002705
7: -0.0130767, -0.0086793, -0.0130671, -0.0085906, -0.0044368, 0.0043395
8: -0.0130706, -0.0048600, -0.0127986, -0.0049214, -0.0065759, 0.0064337
9: -0.0018057, 0.0022809, -0.0017681, 0.0021356, -0.0031431, 0.0032173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016364, upper bound: 0.0016705
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015878, upper bound: 0.0016704
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0026967, 0.0006867, -0.0026691, 0.0006386, -0.0028835, 0.0028979
1: -0.0045510, -0.0034307, -0.0045308, -0.0034533, -0.0010071, 0.0009906
2: 0.0111773, 0.0157333, 0.0112277, 0.0156594, -0.0037853, 0.0038026
3: 1.0069647, 1.0098388, 1.0070105, 1.0097885, -0.0028238, 0.0028284
4: -0.0041969, -0.0034463, -0.0041832, -0.0034571, -0.0006099, 0.0006076
5: 0.0018900, 0.0044986, 0.0019126, 0.0044607, -0.0022148, 0.0022256
6: -0.0025852, -0.0023151, -0.0025863, -0.0023193, -0.0002659, 0.0002711
7: -0.0130764, -0.0087089, -0.0130708, -0.0086514, -0.0043766, 0.0043152
8: -0.0130625, -0.0048961, -0.0129023, -0.0050283, -0.0065408, 0.0065120
9: -0.0017885, 0.0022766, -0.0017153, 0.0021910, -0.0031981, 0.0032029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016028, upper bound: 0.0016853
time: 1.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015990, upper bound: 0.0016701
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027117, 0.0006629, -0.0027117, 0.0006629, -0.0028152, 0.0028152
1: -0.0045552, -0.0034419, -0.0045552, -0.0034419, -0.0009731, 0.0009731
2: 0.0111577, 0.0156968, 0.0111577, 0.0156968, -0.0036788, 0.0036788
3: 1.0069911, 1.0098493, 1.0069911, 1.0098493, -0.0028582, 0.0028582
4: -0.0041901, -0.0034432, -0.0041901, -0.0034432, -0.0005881, 0.0005881
5: 0.0018785, 0.0044798, 0.0018785, 0.0044798, -0.0021608, 0.0021608
6: -0.0025845, -0.0023142, -0.0025845, -0.0023142, -0.0002703, 0.0002703
7: -0.0130736, -0.0086833, -0.0130736, -0.0086833, -0.0043309, 0.0043309
8: -0.0129833, -0.0048635, -0.0129833, -0.0048635, -0.0062876, 0.0062876
9: -0.0018043, 0.0022343, -0.0018043, 0.0022343, -0.0030725, 0.0030725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017020, upper bound: 0.0017318
time: 1.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017021, upper bound: 0.0017286
time: 1.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027117, 0.0006629, -0.0029294, 0.0006169, -0.0028164, 0.0030669
1: -0.0045552, -0.0034419, -0.0045988, -0.0034638, -0.0009791, 0.0010411
2: 0.0111577, 0.0156968, 0.0108971, 0.0156260, -0.0036808, 0.0039994
3: 1.0069911, 1.0098493, 1.0070214, 1.0099580, -0.0029669, 0.0028279
4: -0.0041901, -0.0034432, -0.0041769, -0.0034060, -0.0006376, 0.0005884
5: 0.0018785, 0.0044798, 0.0017135, 0.0044436, -0.0021618, 0.0023534
6: -0.0025845, -0.0023142, -0.0025917, -0.0023053, -0.0002792, 0.0002774
7: -0.0130736, -0.0086833, -0.0130682, -0.0082207, -0.0047962, 0.0043310
8: -0.0129833, -0.0048635, -0.0128299, -0.0044966, -0.0068123, 0.0062918
9: -0.0018043, 0.0022343, -0.0019662, 0.0021523, -0.0030747, 0.0033229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017020, upper bound: 0.0017318
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017021, upper bound: 0.0017286
time: 1.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0029294, 0.0006169, -0.0026704, 0.0006790, -0.0030971, 0.0027755
1: -0.0045988, -0.0034638, -0.0045447, -0.0034345, -0.0010504, 0.0009816
2: 0.0108971, 0.0156260, 0.0112097, 0.0157215, -0.0040457, 0.0036272
3: 1.0070214, 1.0099580, 1.0069755, 1.0098230, -0.0028015, 0.0029825
4: -0.0041769, -0.0034060, -0.0041947, -0.0034512, -0.0005799, 0.0006462
5: 0.0017135, 0.0044436, 0.0019100, 0.0044925, -0.0023771, 0.0021305
6: -0.0025917, -0.0023053, -0.0025839, -0.0023164, -0.0002753, 0.0002786
7: -0.0130682, -0.0082207, -0.0130755, -0.0087559, -0.0042595, 0.0047998
8: -0.0128299, -0.0044966, -0.0130369, -0.0049460, -0.0062030, 0.0069128
9: -0.0019662, 0.0021523, -0.0017653, 0.0022629, -0.0033766, 0.0030315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016670, upper bound: 0.0016993
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016670, upper bound: 0.0016851
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0029139, 0.0006144, -0.0026210, 0.0007167, -0.0031165, 0.0027496
1: -0.0045946, -0.0034650, -0.0045294, -0.0034151, -0.0010659, 0.0009778
2: 0.0109170, 0.0156222, 0.0112755, 0.0157794, -0.0040791, 0.0035999
3: 1.0070252, 1.0099473, 1.0069301, 1.0097851, -0.0027598, 0.0030172
4: -0.0041762, -0.0034091, -0.0042055, -0.0034618, -0.0005766, 0.0006520
5: 0.0017253, 0.0044417, 0.0019480, 0.0045221, -0.0023929, 0.0021112
6: -0.0025910, -0.0023061, -0.0025859, -0.0023195, -0.0002715, 0.0002797
7: -0.0130679, -0.0082472, -0.0130799, -0.0088269, -0.0041883, 0.0047776
8: -0.0128218, -0.0045296, -0.0131624, -0.0050612, -0.0061725, 0.0069743
9: -0.0019505, 0.0021479, -0.0017092, 0.0023300, -0.0034129, 0.0030200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016576, upper bound: 0.0016993
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016576, upper bound: 0.0016851
time: 1.47 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.78 seconds
IS_A1_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016866, upper bound: 0.0016807
IS_A1_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016866, upper bound: 0.0016807
IS_A1_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016866, upper bound: 0.0016729
IS_A1_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016866, upper bound: 0.0016729
IS_A1_B1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016909, upper bound: 0.0015996
IS_A1_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016790, upper bound: 0.0015982
IS_A1_B1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016904, upper bound: 0.0015794
IS_A1_B1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016780, upper bound: 0.0015775
IS_A1_B1_A1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015932, upper bound: 0.0015945
IS_A1_B1_A1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015881, upper bound: 0.0015684
IS_A1_B1_A1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015719, upper bound: 0.0015931
IS_A1_B1_A1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0015656
IS_A1_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017089, upper bound: 0.0016403
IS_A1_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017089, upper bound: 0.0016403
IS_A1_B1_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015719, upper bound: 0.0015923
IS_A1_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016476, upper bound: 0.0015656
IS_A1_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016571, upper bound: 0.0017234
IS_A1_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016432, upper bound: 0.0017234
IS_A1_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016571, upper bound: 0.0017234
IS_A1_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016432, upper bound: 0.0017234
IS_A1_B1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016403, upper bound: 0.0017088
IS_A1_B1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016403, upper bound: 0.0017088
IS_A1_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015923, upper bound: 0.0016490
IS_A1_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016476
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017203
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017176
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017203
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017176
IS_A1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016297, upper bound: 0.0016817
IS_A1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016289, upper bound: 0.0016628
IS_A1_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016195, upper bound: 0.0016816
IS_A1_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016186, upper bound: 0.0016623
IS_A1_B2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016795, upper bound: 0.0017033
IS_A1_B2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016795, upper bound: 0.0017474
IS_A1_B2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016720, upper bound: 0.0017033
IS_A1_B2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016720, upper bound: 0.0017473
IS_A1_B2_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015989, upper bound: 0.0017046
IS_A1_B2_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015980, upper bound: 0.0016929
IS_A1_B2_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015793, upper bound: 0.0017044
IS_A1_B2_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015774, upper bound: 0.0016917
IS_A1_B2_B1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016102, upper bound: 0.0016027
IS_A1_B2_B1_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015878, upper bound: 0.0015988
IS_A1_B2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015926, upper bound: 0.0016962
IS_A1_B2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015878, upper bound: 0.0016773
IS_A1_B2_B1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015923, upper bound: 0.0016015
IS_A1_B2_B1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0015956
IS_A1_B2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015719, upper bound: 0.0016943
IS_A1_B2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016754
IS_A1_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016815
IS_A1_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016692
IS_A1_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016815
IS_A1_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016692
IS_A1_B2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016554, upper bound: 0.0016193
IS_A1_B2_B2_A1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015881, upper bound: 0.0015984
IS_A1_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016490, upper bound: 0.0016181
IS_A1_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016476, upper bound: 0.0015955
IS_A1_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016777, upper bound: 0.0017434
IS_A1_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016707, upper bound: 0.0017434
IS_A1_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016777, upper bound: 0.0017434
IS_A1_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016707, upper bound: 0.0017434
IS_A1_B2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016384, upper bound: 0.0016950
IS_A1_B2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016191, upper bound: 0.0016945
IS_A1_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016384, upper bound: 0.0016908
IS_A1_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016901
IS_A2_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016795
IS_A2_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016795
IS_A2_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016720
IS_A2_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016720
IS_A2_B1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017046, upper bound: 0.0015989
IS_A2_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016929, upper bound: 0.0015980
IS_A2_B1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017044, upper bound: 0.0015793
IS_A2_B1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016918, upper bound: 0.0015774
IS_A2_B1_A1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016027, upper bound: 0.0016102
IS_A2_B1_A1_A2_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015988, upper bound: 0.0015878
IS_A2_B1_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016212, upper bound: 0.0015926
IS_A2_B1_A1_A2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015988, upper bound: 0.0015878
IS_A2_B1_A1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016015, upper bound: 0.0015923
IS_A2_B1_A1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015955, upper bound: 0.0015656
IS_A2_B1_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016194, upper bound: 0.0015719
IS_A2_B1_A1_A2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015955, upper bound: 0.0015656
IS_A2_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016815, upper bound: 0.0017232
IS_A2_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016692, upper bound: 0.0017232
IS_A2_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016815, upper bound: 0.0017232
IS_A2_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016692, upper bound: 0.0017232
IS_A2_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016193, upper bound: 0.0016554
IS_A2_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015984, upper bound: 0.0016542
IS_A2_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016181, upper bound: 0.0016490
IS_A2_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015956, upper bound: 0.0016476
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017190
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017164
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017190
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017164
IS_A2_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016601, upper bound: 0.0016816
IS_A2_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016601, upper bound: 0.0016628
IS_A2_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016515, upper bound: 0.0016815
IS_A2_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016514, upper bound: 0.0016623
IS_A2_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016837
IS_A2_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016837
IS_A2_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016765
IS_A2_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016765
IS_A2_B2_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017099, upper bound: 0.0016109
IS_A2_B2_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017000, upper bound: 0.0016107
IS_A2_B2_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017097, upper bound: 0.0015920
IS_A2_B2_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016780, upper bound: 0.0015917
IS_A2_B2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016221, upper bound: 0.0016059
IS_A2_B2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016200, upper bound: 0.0015842
IS_A2_B2_A1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016028, upper bound: 0.0016047
IS_A2_B2_A1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015990, upper bound: 0.0015819
IS_A2_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016855, upper bound: 0.0016226
IS_A2_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016853, upper bound: 0.0016052
IS_A2_B2_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016999, upper bound: 0.0015852
IS_A2_B2_A1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015990, upper bound: 0.0015819
IS_A2_B2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016820, upper bound: 0.0017307
IS_A2_B2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016695, upper bound: 0.0017307
IS_A2_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016820, upper bound: 0.0017307
IS_A2_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016695, upper bound: 0.0017307
IS_A2_B2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016364, upper bound: 0.0016705
IS_A2_B2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015878, upper bound: 0.0016704
IS_A2_B2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016028, upper bound: 0.0016853
IS_A2_B2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0015990, upper bound: 0.0016701
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017020, upper bound: 0.0017318
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017021, upper bound: 0.0017286
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017020, upper bound: 0.0017318
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0017021, upper bound: 0.0017286
IS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016670, upper bound: 0.0016993
IS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016670, upper bound: 0.0016851
IS_A2_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016576, upper bound: 0.0016993
IS_A2_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.78
Output dim: 3, lower bound: -0.0016576, upper bound: 0.0016851

## BFS IS instance: IS_A1_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0025141, 0.0005828, -0.0025488, 0.0005918, -0.0025688, 0.0025990
1: -0.0045068, -0.0034818, -0.0045161, -0.0034773, -0.0008830, 0.0008898
2: 0.0114038, 0.0155736, 0.0113600, 0.0155874, -0.0033513, 0.0033904
3: 1.0070962, 1.0097286, 1.0070835, 1.0097517, -0.0026555, 0.0026450
4: -0.0041672, -0.0034804, -0.0041697, -0.0034735, -0.0005402, 0.0005344
5: 0.0020292, 0.0044168, 0.0020027, 0.0044238, -0.0019713, 0.0019944
6: -0.0025745, -0.0023242, -0.0025761, -0.0023223, -0.0002522, 0.0002519
7: -0.0130642, -0.0090211, -0.0130653, -0.0089641, -0.0040435, 0.0039874
8: -0.0127163, -0.0052474, -0.0127462, -0.0051747, -0.0057631, 0.0057011
9: -0.0016253, 0.0020916, -0.0016595, 0.0021076, -0.0027773, 0.0028062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016543, upper bound: 0.0016424
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016408, upper bound: 0.0016424
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0025141, 0.0005828, -0.0026848, 0.0006050, -0.0025918, 0.0027322
1: -0.0045068, -0.0034818, -0.0045497, -0.0034705, -0.0008932, 0.0009233
2: 0.0114038, 0.0155736, 0.0111892, 0.0156078, -0.0033866, 0.0035579
3: 1.0070962, 1.0097286, 1.0070637, 1.0098355, -0.0027393, 0.0026648
4: -0.0041672, -0.0034804, -0.0041735, -0.0034477, -0.0005658, 0.0005410
5: 0.0020292, 0.0044168, 0.0018989, 0.0044343, -0.0019894, 0.0020963
6: -0.0025745, -0.0023242, -0.0025806, -0.0023154, -0.0002592, 0.0002564
7: -0.0130642, -0.0090211, -0.0130668, -0.0087306, -0.0042766, 0.0039901
8: -0.0127163, -0.0052474, -0.0127904, -0.0049096, -0.0060286, 0.0057776
9: -0.0016253, 0.0020916, -0.0017836, 0.0021312, -0.0028181, 0.0029294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016543, upper bound: 0.0016424
time: 1.55 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016408, upper bound: 0.0016424
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0024596, 0.0006193, -0.0025341, 0.0005898, -0.0025445, 0.0026609
1: -0.0044914, -0.0034635, -0.0045122, -0.0034783, -0.0008806, 0.0009227
2: 0.0114749, 0.0156297, 0.0113786, 0.0155844, -0.0033268, 0.0034900
3: 1.0070525, 1.0096903, 1.0070868, 1.0097421, -0.0026896, 0.0026035
4: -0.0041776, -0.0034915, -0.0041692, -0.0034764, -0.0005594, 0.0005315
5: 0.0020709, 0.0044455, 0.0020139, 0.0044223, -0.0019532, 0.0020435
6: -0.0025762, -0.0023274, -0.0025755, -0.0023231, -0.0002531, 0.0002482
7: -0.0130685, -0.0091083, -0.0130650, -0.0089897, -0.0040270, 0.0039053
8: -0.0128378, -0.0053646, -0.0127398, -0.0052047, -0.0059907, 0.0056781
9: -0.0015689, 0.0021565, -0.0016454, 0.0021041, -0.0027699, 0.0029297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016541, upper bound: 0.0016322
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016401, upper bound: 0.0016322
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0024596, 0.0006193, -0.0026703, 0.0006032, -0.0025676, 0.0027950
1: -0.0044914, -0.0034635, -0.0045458, -0.0034715, -0.0008909, 0.0009562
2: 0.0114749, 0.0156297, 0.0112077, 0.0156050, -0.0033624, 0.0036578
3: 1.0070525, 1.0096903, 1.0070667, 1.0098259, -0.0027734, 0.0026236
4: -0.0041776, -0.0034915, -0.0041730, -0.0034506, -0.0005850, 0.0005382
5: 0.0020709, 0.0044455, 0.0019100, 0.0044328, -0.0019715, 0.0021460
6: -0.0025762, -0.0023274, -0.0025800, -0.0023162, -0.0002600, 0.0002527
7: -0.0130685, -0.0091083, -0.0130666, -0.0087545, -0.0042613, 0.0039080
8: -0.0128378, -0.0053646, -0.0127843, -0.0049397, -0.0062566, 0.0057554
9: -0.0015689, 0.0021565, -0.0017694, 0.0021280, -0.0028112, 0.0030532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016541, upper bound: 0.0016322
time: 1.61 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016401, upper bound: 0.0016322
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0025131, 0.0005446, -0.0028679, 0.0005541, -0.0025973, 0.0029016
1: -0.0045066, -0.0035008, -0.0045864, -0.0034941, -0.0009028, 0.0009582
2: 0.0114050, 0.0155148, 0.0109718, 0.0155295, -0.0033954, 0.0037591
3: 1.0071406, 1.0097283, 1.0070959, 1.0099272, -0.0027865, 0.0026324
4: -0.0041562, -0.0034805, -0.0041590, -0.0034167, -0.0005942, 0.0005427
5: 0.0020300, 0.0043867, 0.0017601, 0.0043942, -0.0019938, 0.0022245
6: -0.0025725, -0.0023242, -0.0025873, -0.0023078, -0.0002646, 0.0002631
7: -0.0130597, -0.0090234, -0.0130608, -0.0083257, -0.0046825, 0.0039886
8: -0.0125890, -0.0052487, -0.0126208, -0.0046046, -0.0063099, 0.0057984
9: -0.0016248, 0.0020236, -0.0019199, 0.0020406, -0.0028298, 0.0030573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016909, upper bound: 0.0015996
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016909, upper bound: 0.0015996
time: 1.52 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0025990, 0.0004926, -0.0028676, 0.0005378, -0.0027070, 0.0028991
1: -0.0045233, -0.0035246, -0.0045864, -0.0035020, -0.0009374, 0.0009702
2: 0.0113050, 0.0154350, 0.0109721, 0.0155045, -0.0035374, 0.0037553
3: 1.0071841, 1.0097697, 1.0071138, 1.0099270, -0.0027429, 0.0026559
4: -0.0041414, -0.0034663, -0.0041543, -0.0034168, -0.0005935, 0.0005659
5: 0.0019651, 0.0043458, 0.0017604, 0.0043814, -0.0020777, 0.0022225
6: -0.0025750, -0.0023208, -0.0025866, -0.0023078, -0.0002672, 0.0002658
7: -0.0130536, -0.0088073, -0.0130589, -0.0083264, -0.0046815, 0.0042061
8: -0.0124159, -0.0051085, -0.0125665, -0.0046051, -0.0063022, 0.0060532
9: -0.0016865, 0.0019311, -0.0019198, 0.0020115, -0.0029552, 0.0030535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016790, upper bound: 0.0015982
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016790, upper bound: 0.0015982
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0024586, 0.0005810, -0.0028540, 0.0005523, -0.0025731, 0.0029661
1: -0.0044913, -0.0034828, -0.0045826, -0.0034951, -0.0009005, 0.0009913
2: 0.0114761, 0.0155708, 0.0109897, 0.0155267, -0.0033712, 0.0038618
3: 1.0070972, 1.0096899, 1.0070990, 1.0099176, -0.0028204, 0.0025909
4: -0.0041666, -0.0034916, -0.0041584, -0.0034195, -0.0006139, 0.0005399
5: 0.0020718, 0.0044153, 0.0017708, 0.0043927, -0.0019759, 0.0022755
6: -0.0025740, -0.0023274, -0.0025868, -0.0023086, -0.0002654, 0.0002594
7: -0.0130640, -0.0091105, -0.0130606, -0.0083486, -0.0046680, 0.0039064
8: -0.0127102, -0.0053659, -0.0126146, -0.0046337, -0.0065446, 0.0057760
9: -0.0015683, 0.0020883, -0.0019060, 0.0020373, -0.0028228, 0.0031834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016130, upper bound: 0.0015794
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016130, upper bound: 0.0015794
time: 1.50 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0025468, 0.0005309, -0.0028536, 0.0005360, -0.0026839, 0.0029655
1: -0.0045075, -0.0035055, -0.0045826, -0.0035030, -0.0009351, 0.0010031
2: 0.0113739, 0.0154938, 0.0109901, 0.0155016, -0.0035127, 0.0038610
3: 1.0071409, 1.0097306, 1.0071168, 1.0099176, -0.0027767, 0.0026138
4: -0.0041523, -0.0034773, -0.0041538, -0.0034195, -0.0006138, 0.0005631
5: 0.0020052, 0.0043759, 0.0017711, 0.0043799, -0.0020606, 0.0022750
6: -0.0025764, -0.0023240, -0.0025861, -0.0023086, -0.0002678, 0.0002620
7: -0.0130581, -0.0088928, -0.0130587, -0.0083493, -0.0046673, 0.0041258
8: -0.0125435, -0.0052264, -0.0125603, -0.0046342, -0.0065434, 0.0060314
9: -0.0016288, 0.0019992, -0.0019058, 0.0020082, -0.0029482, 0.0031830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015929, upper bound: 0.0015775
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015929, upper bound: 0.0015775
time: 1.51 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.45 seconds
IS_A1_B1_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016543, upper bound: 0.0016424
IS_A1_B1_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016408, upper bound: 0.0016424
IS_A1_B1_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016543, upper bound: 0.0016424
IS_A1_B1_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016408, upper bound: 0.0016424
IS_A1_B1_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016541, upper bound: 0.0016322
IS_A1_B1_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016401, upper bound: 0.0016322
IS_A1_B1_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016541, upper bound: 0.0016322
IS_A1_B1_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016401, upper bound: 0.0016322
IS_A1_B1_A1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016909, upper bound: 0.0015996
IS_A1_B1_A1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016909, upper bound: 0.0015996
IS_A1_B1_A1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016790, upper bound: 0.0015982
IS_A1_B1_A1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016790, upper bound: 0.0015982
IS_A1_B1_A1_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016130, upper bound: 0.0015794
IS_A1_B1_A1_A1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0016130, upper bound: 0.0015794
IS_A1_B1_A1_A1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0015929, upper bound: 0.0015775
IS_A1_B1_A1_A1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.45
Output dim: 3, lower bound: -0.0015929, upper bound: 0.0015775
IS_A1_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017089, upper bound: 0.0016403
IS_A1_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017089, upper bound: 0.0016403
IS_A1_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016476, upper bound: 0.0015656
IS_A1_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016571, upper bound: 0.0017234
IS_A1_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016432, upper bound: 0.0017234
IS_A1_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016571, upper bound: 0.0017234
IS_A1_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016432, upper bound: 0.0017234
IS_A1_B1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016403, upper bound: 0.0017088
IS_A1_B1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016403, upper bound: 0.0017088
IS_A1_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015923, upper bound: 0.0016490
IS_A1_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016476
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017203
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017176
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017203
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016792, upper bound: 0.0017176
IS_A1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016297, upper bound: 0.0016817
IS_A1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016289, upper bound: 0.0016628
IS_A1_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016195, upper bound: 0.0016816
IS_A1_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016186, upper bound: 0.0016623
IS_A1_B2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016795, upper bound: 0.0017033
IS_A1_B2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016795, upper bound: 0.0017474
IS_A1_B2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016720, upper bound: 0.0017033
IS_A1_B2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016720, upper bound: 0.0017473
IS_A1_B2_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015989, upper bound: 0.0017046
IS_A1_B2_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015980, upper bound: 0.0016929
IS_A1_B2_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015793, upper bound: 0.0017044
IS_A1_B2_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015774, upper bound: 0.0016917
IS_A1_B2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015926, upper bound: 0.0016962
IS_A1_B2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015878, upper bound: 0.0016773
IS_A1_B2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015719, upper bound: 0.0016943
IS_A1_B2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016754
IS_A1_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016815
IS_A1_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016692
IS_A1_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016815
IS_A1_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017232, upper bound: 0.0016692
IS_A1_B2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016554, upper bound: 0.0016193
IS_A1_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016490, upper bound: 0.0016181
IS_A1_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016476, upper bound: 0.0015955
IS_A1_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016777, upper bound: 0.0017434
IS_A1_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016707, upper bound: 0.0017434
IS_A1_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016777, upper bound: 0.0017434
IS_A1_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016707, upper bound: 0.0017434
IS_A1_B2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016384, upper bound: 0.0016950
IS_A1_B2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016191, upper bound: 0.0016945
IS_A1_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016384, upper bound: 0.0016908
IS_A1_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016901
IS_A2_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016795
IS_A2_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016795
IS_A2_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016720
IS_A2_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017033, upper bound: 0.0016720
IS_A2_B1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017046, upper bound: 0.0015989
IS_A2_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016929, upper bound: 0.0015980
IS_A2_B1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017044, upper bound: 0.0015793
IS_A2_B1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016918, upper bound: 0.0015774
IS_A2_B1_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016212, upper bound: 0.0015926
IS_A2_B1_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016194, upper bound: 0.0015719
IS_A2_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016815, upper bound: 0.0017232
IS_A2_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016692, upper bound: 0.0017232
IS_A2_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016815, upper bound: 0.0017232
IS_A2_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016692, upper bound: 0.0017232
IS_A2_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016193, upper bound: 0.0016554
IS_A2_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015984, upper bound: 0.0016542
IS_A2_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016181, upper bound: 0.0016490
IS_A2_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015956, upper bound: 0.0016476
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017190
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017164
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017190
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016970, upper bound: 0.0017164
IS_A2_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016601, upper bound: 0.0016816
IS_A2_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016601, upper bound: 0.0016628
IS_A2_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016515, upper bound: 0.0016815
IS_A2_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016514, upper bound: 0.0016623
IS_A2_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016837
IS_A2_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016837
IS_A2_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016765
IS_A2_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017041, upper bound: 0.0016765
IS_A2_B2_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017099, upper bound: 0.0016109
IS_A2_B2_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017000, upper bound: 0.0016107
IS_A2_B2_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017097, upper bound: 0.0015920
IS_A2_B2_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016780, upper bound: 0.0015917
IS_A2_B2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016221, upper bound: 0.0016059
IS_A2_B2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016200, upper bound: 0.0015842
IS_A2_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016855, upper bound: 0.0016226
IS_A2_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016853, upper bound: 0.0016052
IS_A2_B2_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016999, upper bound: 0.0015852
IS_A2_B2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016820, upper bound: 0.0017307
IS_A2_B2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016695, upper bound: 0.0017307
IS_A2_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016820, upper bound: 0.0017307
IS_A2_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016695, upper bound: 0.0017307
IS_A2_B2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016364, upper bound: 0.0016705
IS_A2_B2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015878, upper bound: 0.0016704
IS_A2_B2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016028, upper bound: 0.0016853
IS_A2_B2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0015990, upper bound: 0.0016701
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017020, upper bound: 0.0017318
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017021, upper bound: 0.0017286
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017020, upper bound: 0.0017318
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0017021, upper bound: 0.0017286
IS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016670, upper bound: 0.0016993
IS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016670, upper bound: 0.0016851
IS_A2_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016576, upper bound: 0.0016993
IS_A2_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 3, lower bound: -0.0016576, upper bound: 0.0016851

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.97 + 598.07 = 602.03 seconds
