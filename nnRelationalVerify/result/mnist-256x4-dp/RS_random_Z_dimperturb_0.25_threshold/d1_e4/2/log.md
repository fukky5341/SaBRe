## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 6.43e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009888, 0.0009888)
1: (-0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002173, 0.0002173)
2: (0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008883, 0.0008883)
3: (1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266)
4: (-0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001332, 0.0001332)
5: (0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008639, 0.0008639)
6: (-0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500)
7: (-0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020684, 0.0020684)
8: (-0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0011013, 0.0011013)
9: (0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005274, 0.0005274)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 1.44 = 3.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0001393, upper bound: 0.0001393

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001241, upper bound: 0.0001241
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001241, upper bound: 0.0001241
time: 0.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.26 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 3, lower bound: -0.0001241, upper bound: 0.0001241
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 3, lower bound: -0.0001241, upper bound: 0.0001241

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009822, 0.0009830
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002148, 0.0002152
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008776, 0.0008790
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001313, 0.0001311
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008587, 0.0008594
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020678, 0.0020677
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010818, 0.0010789
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005142, 0.0005157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001137, upper bound: 0.0001106
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001108, upper bound: 0.0001137
time: 0.55 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009888, 0.0009822
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002173, 0.0002148
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008883, 0.0008776
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001311, 0.0001332
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008639, 0.0008587
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020677, 0.0020684
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010789, 0.0011013
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005274, 0.0005142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001205, upper bound: 0.0001149
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001148, upper bound: 0.0001205
time: 0.54 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 3, lower bound: -0.0001137, upper bound: 0.0001106
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 3, lower bound: -0.0001108, upper bound: 0.0001137
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 3, lower bound: -0.0001205, upper bound: 0.0001149
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 3, lower bound: -0.0001148, upper bound: 0.0001205

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009819, 0.0009828
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002147, 0.0002151
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008772, 0.0008785
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001312, 0.0001310
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008585, 0.0008592
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020677, 0.0020676
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010810, 0.0010780
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005135, 0.0005151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001135, upper bound: 0.0001077
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001092, upper bound: 0.0001104
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009819, 0.0009828
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002147, 0.0002151
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008772, 0.0008785
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001312, 0.0001310
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008585, 0.0008591
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020677, 0.0020676
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010809, 0.0010781
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005136, 0.0005151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001073, upper bound: 0.0001045
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001008, upper bound: 0.0001101
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009866, 0.0009804
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002168, 0.0002145
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008849, 0.0008750
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001307, 0.0001327
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008622, 0.0008573
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020675, 0.0020682
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010724, 0.0010934
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005236, 0.0005113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001202, upper bound: 0.0001122
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001157, upper bound: 0.0001146
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009871, 0.0009800
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002170, 0.0002143
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008856, 0.0008743
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001306, 0.0001329
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008625, 0.0008569
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020674, 0.0020682
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010709, 0.0010949
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005244, 0.0005105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001000, upper bound: 0.0001044
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001006, upper bound: 0.0001043
time: 0.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 3, lower bound: -0.0001135, upper bound: 0.0001077
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 3, lower bound: -0.0001092, upper bound: 0.0001104
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 3, lower bound: -0.0001073, upper bound: 0.0001045
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 3, lower bound: -0.0001008, upper bound: 0.0001101
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 3, lower bound: -0.0001202, upper bound: 0.0001122
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 3, lower bound: -0.0001157, upper bound: 0.0001146
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 3, lower bound: -0.0001000, upper bound: 0.0001044
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 3, lower bound: -0.0001006, upper bound: 0.0001043

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009601, 0.0009659
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002074, 0.0002102
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008430, 0.0008517
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001273, 0.0001257
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008413, 0.0008458
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020658, 0.0020652
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010183, 0.0009993
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004780, 0.0004881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001115, upper bound: 0.0001027
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001079, upper bound: 0.0001056
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009639, 0.0009610
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002092, 0.0002078
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008488, 0.0008443
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001259, 0.0001268
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008443, 0.0008420
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020653, 0.0020656
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010023, 0.0010119
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004847, 0.0004796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001007, upper bound: 0.0000972
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000992, upper bound: 0.0001009
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009797, 0.0009810
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002142, 0.0002148
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008739, 0.0008759
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001309, 0.0001305
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008567, 0.0008577
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020675, 0.0020674
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010744, 0.0010701
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005100, 0.0005122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000838, upper bound: 0.0000806
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000838, upper bound: 0.0000806
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009802, 0.0009806
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002144, 0.0002146
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008746, 0.0008752
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001308, 0.0001306
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008571, 0.0008574
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020675, 0.0020674
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010729, 0.0010716
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005108, 0.0005114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000885, upper bound: 0.0000960
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000957
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009646, 0.0009624
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002093, 0.0002091
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008506, 0.0008466
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001267, 0.0001273
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008448, 0.0008431
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020654, 0.0020657
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010075, 0.0010162
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004863, 0.0004839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001103, upper bound: 0.0001033
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001057, upper bound: 0.0001044
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009684, 0.0009587
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002111, 0.0002073
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008564, 0.0008408
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001256, 0.0001283
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008478, 0.0008401
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020650, 0.0020661
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009950, 0.0010288
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004930, 0.0004772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001137, upper bound: 0.0001111
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0001124, upper bound: 0.0001126
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009859, 0.0009640
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002168, 0.0002069
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008847, 0.0008506
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001263, 0.0001328
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008616, 0.0008443
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020655, 0.0020681
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010220, 0.0010953
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005241, 0.0004838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000981, upper bound: 0.0000990
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000971, upper bound: 0.0001024
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009711, 0.0009800
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002096, 0.0002143
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008619, 0.0008743
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001306, 0.0001285
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008499, 0.0008569
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020674, 0.0020663
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010709, 0.0010460
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004977, 0.0005105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000987, upper bound: 0.0000988
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000974, upper bound: 0.0001022
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0001115, upper bound: 0.0001027
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0001079, upper bound: 0.0001056
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0001007, upper bound: 0.0000972
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0000992, upper bound: 0.0001009
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0000838, upper bound: 0.0000806
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0000838, upper bound: 0.0000806
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0000885, upper bound: 0.0000960
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0000883, upper bound: 0.0000957
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0001103, upper bound: 0.0001033
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0001057, upper bound: 0.0001044
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0001137, upper bound: 0.0001111
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0001124, upper bound: 0.0001126
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0000981, upper bound: 0.0000990
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0000971, upper bound: 0.0001024
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0000987, upper bound: 0.0000988
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 3, lower bound: -0.0000974, upper bound: 0.0001022

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009573, 0.0009636
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002072, 0.0002103
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008387, 0.0008483
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001270, 0.0001252
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008391, 0.0008440
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020655, 0.0020648
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010122, 0.0009912
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004763, 0.0004875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000973, upper bound: 0.0000892
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000976, upper bound: 0.0000892
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009577, 0.0009630
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002074, 0.0002100
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008393, 0.0008475
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001269, 0.0001253
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008394, 0.0008436
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020655, 0.0020649
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010102, 0.0009926
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004770, 0.0004864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000829, upper bound: 0.0000849
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000829, upper bound: 0.0000849
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009601, 0.0009588
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002085, 0.0002079
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008432, 0.0008412
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001260, 0.0001264
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008412, 0.0008402
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020650, 0.0020651
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009962, 0.0010004
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004819, 0.0004796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000990, upper bound: 0.0000931
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000978, upper bound: 0.0000954
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009617, 0.0009572
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002093, 0.0002071
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008457, 0.0008387
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001255, 0.0001268
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008425, 0.0008389
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020648, 0.0020653
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009908, 0.0010059
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004848, 0.0004768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000856, upper bound: 0.0000873
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000856, upper bound: 0.0000873
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009788, 0.0009651
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002141, 0.0002074
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008733, 0.0008522
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001266, 0.0001305
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008560, 0.0008451
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020657, 0.0020673
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010255, 0.0010712
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005100, 0.0004855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000698, upper bound: 0.0000678
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000679
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009638, 0.0009810
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002068, 0.0002148
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008502, 0.0008759
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001309, 0.0001262
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008441, 0.0008577
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020675, 0.0020655
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010744, 0.0010212
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004832, 0.0005122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000669
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000669
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009118, 0.0009034
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002156, 0.0002115
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007617, 0.0007489
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001213, 0.0001237
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008035, 0.0007970
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020585, 0.0020595
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008183, 0.0008461
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004853, 0.0004705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000728
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000728
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009030, 0.0009128
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002113, 0.0002161
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007483, 0.0007634
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001240, 0.0001212
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007966, 0.0008043
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020596, 0.0020584
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008496, 0.0008170
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004698, 0.0004872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000699
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000699
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009607, 0.0009602
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002086, 0.0002092
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008450, 0.0008435
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001268, 0.0001269
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008418, 0.0008413
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020652, 0.0020652
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010015, 0.0010048
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004839, 0.0004841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000961, upper bound: 0.0000904
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000963, upper bound: 0.0000904
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009623, 0.0009586
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002093, 0.0002084
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008475, 0.0008410
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001263, 0.0001273
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008430, 0.0008400
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020650, 0.0020654
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009960, 0.0010101
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004867, 0.0004811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000922, upper bound: 0.0000913
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000921, upper bound: 0.0000914
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009656, 0.0009563
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002108, 0.0002073
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008522, 0.0008372
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001253, 0.0001279
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008456, 0.0008383
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020647, 0.0020658
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009887, 0.0010211
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004912, 0.0004765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000960, upper bound: 0.0000950
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000962, upper bound: 0.0000949
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009661, 0.0009559
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002111, 0.0002071
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008529, 0.0008366
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001252, 0.0001280
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008460, 0.0008379
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020647, 0.0020658
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009873, 0.0010227
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004921, 0.0004757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000988, upper bound: 0.0000997
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000989, upper bound: 0.0000997
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009828, 0.0009613
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002162, 0.0002065
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008802, 0.0008468
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001257, 0.0001321
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008592, 0.0008422
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020652, 0.0020677
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010147, 0.0010869
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005203, 0.0004810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000786, upper bound: 0.0000781
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000786
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009831, 0.0009609
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002163, 0.0002063
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008805, 0.0008462
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001256, 0.0001322
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008594, 0.0008419
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020652, 0.0020677
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010133, 0.0010876
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005206, 0.0004803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000968, upper bound: 0.0000984
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000952, upper bound: 0.0001021
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009680, 0.0009775
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002089, 0.0002138
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008574, 0.0008706
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001301, 0.0001279
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008475, 0.0008550
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020671, 0.0020660
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010638, 0.0010376
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004939, 0.0005073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000856, upper bound: 0.0000850
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000856, upper bound: 0.0000850
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009685, 0.0009771
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002091, 0.0002136
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008581, 0.0008700
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001300, 0.0001280
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008479, 0.0008547
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020671, 0.0020660
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010625, 0.0010390
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004947, 0.0005066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000842, upper bound: 0.0000880
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000842, upper bound: 0.0000877
time: 0.70 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000973, upper bound: 0.0000892
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000976, upper bound: 0.0000892
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000829, upper bound: 0.0000849
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000829, upper bound: 0.0000849
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000990, upper bound: 0.0000931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000978, upper bound: 0.0000954
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000856, upper bound: 0.0000873
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000856, upper bound: 0.0000873
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000698, upper bound: 0.0000678
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000699, upper bound: 0.0000679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000669
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000669
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000699
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000699
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000961, upper bound: 0.0000904
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000963, upper bound: 0.0000904
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000922, upper bound: 0.0000913
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000921, upper bound: 0.0000914
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000960, upper bound: 0.0000950
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000962, upper bound: 0.0000949
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000988, upper bound: 0.0000997
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000989, upper bound: 0.0000997
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000786, upper bound: 0.0000781
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000784, upper bound: 0.0000786
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000968, upper bound: 0.0000984
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000952, upper bound: 0.0001021
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000856, upper bound: 0.0000850
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000856, upper bound: 0.0000850
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000842, upper bound: 0.0000880
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 3, lower bound: -0.0000842, upper bound: 0.0000877

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008875, 0.0008851
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002062, 0.0002050
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007229, 0.0007192
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001179, 0.0001186
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007844, 0.0007825
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020565, 0.0020568
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007203, 0.0007283
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004411, 0.0004368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000721, upper bound: 0.0000688
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000721, upper bound: 0.0000688
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008788, 0.0008941
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002019, 0.0002094
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007095, 0.0007330
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001204, 0.0001161
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007775, 0.0007896
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020575, 0.0020558
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007503, 0.0006994
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004256, 0.0004528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000722, upper bound: 0.0000687
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000722, upper bound: 0.0000687
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009564, 0.0009469
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002067, 0.0002021
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008377, 0.0008231
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001224, 0.0001251
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008382, 0.0008307
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020636, 0.0020647
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009605, 0.0009922
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004765, 0.0004596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000700
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000701
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009416, 0.0009630
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001995, 0.0002100
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008150, 0.0008475
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001269, 0.0001209
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008266, 0.0008436
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020655, 0.0020630
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010102, 0.0009429
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004502, 0.0004864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000701
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000701
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009572, 0.0009565
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002083, 0.0002079
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008389, 0.0008378
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001257, 0.0001259
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008390, 0.0008385
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020647, 0.0020648
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009900, 0.0009923
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004802, 0.0004789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000948, upper bound: 0.0000871
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000887
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009577, 0.0009560
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002085, 0.0002076
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008397, 0.0008370
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001255, 0.0001260
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008394, 0.0008380
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020646, 0.0020648
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009881, 0.0009940
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004810, 0.0004779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000933, upper bound: 0.0000884
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000917, upper bound: 0.0000913
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008915, 0.0008786
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002081, 0.0002018
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007293, 0.0007094
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001163, 0.0001199
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007875, 0.0007774
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020557, 0.0020573
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006984, 0.0007414
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004485, 0.0004255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000838, upper bound: 0.0000809
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000824, upper bound: 0.0000856
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008831, 0.0008880
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002040, 0.0002064
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007164, 0.0007238
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001189, 0.0001176
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007809, 0.0007847
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020568, 0.0020563
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007296, 0.0007135
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004336, 0.0004422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000608, upper bound: 0.0000610
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000608, upper bound: 0.0000610
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009078, 0.0008870
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002133, 0.0002031
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007572, 0.0007253
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001169, 0.0001228
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008004, 0.0007840
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020567, 0.0020591
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007656, 0.0008349
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004765, 0.0004395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000695, upper bound: 0.0000667
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000679, upper bound: 0.0000676
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009008, 0.0008964
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002098, 0.0002077
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007464, 0.0007396
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001195, 0.0001208
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007948, 0.0007914
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020578, 0.0020583
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007968, 0.0008114
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004640, 0.0004562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000696, upper bound: 0.0000667
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000682, upper bound: 0.0000677
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009598, 0.0009787
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002059, 0.0002147
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008442, 0.0008724
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001308, 0.0001256
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008410, 0.0008559
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020673, 0.0020650
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010668, 0.0010081
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004794, 0.0005113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000544, upper bound: 0.0000539
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000540
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009614, 0.0009770
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002066, 0.0002139
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008467, 0.0008699
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001303, 0.0001261
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008423, 0.0008546
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020671, 0.0020652
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010613, 0.0010133
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004822, 0.0005084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000674, upper bound: 0.0000667
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000675, upper bound: 0.0000667
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009117, 0.0009027
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002157, 0.0002113
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007617, 0.0007478
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001211, 0.0001237
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008035, 0.0007964
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020584, 0.0020595
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008167, 0.0008468
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004861, 0.0004700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000699
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000673, upper bound: 0.0000726
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009110, 0.0009034
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002153, 0.0002115
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007606, 0.0007489
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001213, 0.0001235
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008029, 0.0007970
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020585, 0.0020594
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008183, 0.0008444
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004848, 0.0004705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000567, upper bound: 0.0000576
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000566, upper bound: 0.0000581
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009010, 0.0008959
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002099, 0.0002075
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007467, 0.0007390
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001194, 0.0001209
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007950, 0.0007910
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020577, 0.0020583
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007954, 0.0008122
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004644, 0.0004554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000650
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000638, upper bound: 0.0000678
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008862, 0.0009128
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002027, 0.0002161
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007240, 0.0007634
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001240, 0.0001166
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007833, 0.0008043
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020596, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008496, 0.0007628
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004380, 0.0004872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000674, upper bound: 0.0000679
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000667, upper bound: 0.0000696
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008909, 0.0008816
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002071, 0.0002039
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007291, 0.0007141
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001175, 0.0001197
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007870, 0.0007797
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020561, 0.0020572
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007085, 0.0007425
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004467, 0.0004339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000702
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000702
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008822, 0.0008900
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002029, 0.0002080
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007158, 0.0007271
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001199, 0.0001172
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007802, 0.0007863
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020571, 0.0020562
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007364, 0.0007136
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004312, 0.0004489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000702
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000702
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008924, 0.0008799
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002079, 0.0002031
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007315, 0.0007116
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001171, 0.0001201
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007882, 0.0007784
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020559, 0.0020574
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007029, 0.0007476
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004494, 0.0004310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000816, upper bound: 0.0000792
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000797, upper bound: 0.0000809
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008838, 0.0008885
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002036, 0.0002073
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007182, 0.0007247
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001195, 0.0001177
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007814, 0.0007851
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020569, 0.0020564
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007313, 0.0007188
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004340, 0.0004461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000709
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000709
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009647, 0.0009555
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002104, 0.0002069
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008507, 0.0008360
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001251, 0.0001276
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008449, 0.0008376
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020646, 0.0020657
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009860, 0.0010180
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004897, 0.0004752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000820, upper bound: 0.0000783
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000818, upper bound: 0.0000790
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009648, 0.0009563
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002104, 0.0002073
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008509, 0.0008372
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001253, 0.0001276
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008450, 0.0008383
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020647, 0.0020657
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009887, 0.0010184
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004899, 0.0004765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000852, upper bound: 0.0000827
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0000831
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008960, 0.0008773
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002097, 0.0002019
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007367, 0.0007072
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001161, 0.0001209
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007911, 0.0007764
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020556, 0.0020578
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006945, 0.0007592
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004552, 0.0004261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000901, upper bound: 0.0000889
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000886, upper bound: 0.0000910
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008876, 0.0008860
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002056, 0.0002061
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007237, 0.0007206
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001185, 0.0001184
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007844, 0.0007832
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020566, 0.0020568
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007236, 0.0007311
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004401, 0.0004416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000901, upper bound: 0.0000889
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000886, upper bound: 0.0000910
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009825, 0.0009611
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002160, 0.0002064
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008797, 0.0008464
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001256, 0.0001320
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008590, 0.0008420
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020652, 0.0020677
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010139, 0.0010859
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005196, 0.0004804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000783, upper bound: 0.0000778
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000775, upper bound: 0.0000779
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009826, 0.0009611
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002161, 0.0002063
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008798, 0.0008463
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001256, 0.0001320
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008590, 0.0008420
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020652, 0.0020677
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010138, 0.0010861
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0005197, 0.0004804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000650
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000650
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009612, 0.0009431
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002086, 0.0002008
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008459, 0.0008174
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001218, 0.0001268
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008420, 0.0008277
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020632, 0.0020653
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009488, 0.0010105
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004854, 0.0004550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000836, upper bound: 0.0000844
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000837, upper bound: 0.0000841
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009640, 0.0009393
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002100, 0.0001990
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008501, 0.0008116
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001207, 0.0001276
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008442, 0.0008248
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020627, 0.0020656
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009362, 0.0010197
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004903, 0.0004483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000815
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000764, upper bound: 0.0000816
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008985, 0.0009001
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002082, 0.0002106
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007435, 0.0007443
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001208, 0.0001203
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007930, 0.0007944
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020581, 0.0020580
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008047, 0.0008044
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004596, 0.0004666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000660, upper bound: 0.0000645
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000650
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008897, 0.0009089
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002040, 0.0002149
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007301, 0.0007577
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001232, 0.0001178
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007861, 0.0008012
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020591, 0.0020570
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008338, 0.0007754
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004441, 0.0004821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000689
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000689
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008989, 0.0008997
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002084, 0.0002104
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007441, 0.0007437
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001206, 0.0001204
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007933, 0.0007940
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020581, 0.0020581
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008034, 0.0008058
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004604, 0.0004659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000700
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000700
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008902, 0.0009084
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002042, 0.0002147
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007308, 0.0007571
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001231, 0.0001179
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007865, 0.0008009
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020591, 0.0020570
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008323, 0.0007768
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004448, 0.0004814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000698
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000699
time: 0.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000721, upper bound: 0.0000688
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000721, upper bound: 0.0000688
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000722, upper bound: 0.0000687
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000722, upper bound: 0.0000687
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000700
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000701
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000701
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000701
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000948, upper bound: 0.0000871
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000926, upper bound: 0.0000887
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000933, upper bound: 0.0000884
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000917, upper bound: 0.0000913
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000838, upper bound: 0.0000809
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000824, upper bound: 0.0000856
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000608, upper bound: 0.0000610
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000608, upper bound: 0.0000610
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000695, upper bound: 0.0000667
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000679, upper bound: 0.0000676
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000696, upper bound: 0.0000667
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000682, upper bound: 0.0000677
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000544, upper bound: 0.0000539
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000540
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000674, upper bound: 0.0000667
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000675, upper bound: 0.0000667
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000699
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000673, upper bound: 0.0000726
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000567, upper bound: 0.0000576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000566, upper bound: 0.0000581
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000650
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000638, upper bound: 0.0000678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000674, upper bound: 0.0000679
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000667, upper bound: 0.0000696
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000702
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000816, upper bound: 0.0000792
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000797, upper bound: 0.0000809
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000709
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000720, upper bound: 0.0000709
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000820, upper bound: 0.0000783
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000818, upper bound: 0.0000790
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000852, upper bound: 0.0000827
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000847, upper bound: 0.0000831
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000901, upper bound: 0.0000889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000886, upper bound: 0.0000910
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000901, upper bound: 0.0000889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000886, upper bound: 0.0000910
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000783, upper bound: 0.0000778
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000775, upper bound: 0.0000779
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000650
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000650
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000836, upper bound: 0.0000844
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000837, upper bound: 0.0000841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000766, upper bound: 0.0000815
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000764, upper bound: 0.0000816
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000660, upper bound: 0.0000645
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000650
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000689
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000689, upper bound: 0.0000689
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000700
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000700
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000698
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000699

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008842, 0.0008684
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002034, 0.0001957
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007178, 0.0006935
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001129, 0.0001175
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007818, 0.0007694
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020546, 0.0020565
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006596, 0.0007122
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004223, 0.0003942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000634
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000642
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008708, 0.0008851
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001969, 0.0002050
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006972, 0.0007192
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001179, 0.0001136
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007712, 0.0007825
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020565, 0.0020549
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007203, 0.0006676
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003985, 0.0004368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000634
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000642
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008771, 0.0008774
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002000, 0.0002001
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007069, 0.0007073
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001155, 0.0001154
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007762, 0.0007764
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020557, 0.0020556
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006896, 0.0006887
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004097, 0.0004102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000677, upper bound: 0.0000634
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000641
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008621, 0.0008941
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001926, 0.0002094
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006838, 0.0007330
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001204, 0.0001111
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007644, 0.0007896
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020575, 0.0020539
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007503, 0.0006387
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003830, 0.0004528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000677, upper bound: 0.0000634
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000641
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009526, 0.0009446
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002060, 0.0002021
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008322, 0.0008200
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001225, 0.0001247
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008352, 0.0008290
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020633, 0.0020643
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009543, 0.0009807
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004737, 0.0004596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000645, upper bound: 0.0000650
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000657
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009537, 0.0009431
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002065, 0.0002013
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008339, 0.0008176
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001220, 0.0001250
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008361, 0.0008277
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020631, 0.0020644
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009490, 0.0009845
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004757, 0.0004568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000645, upper bound: 0.0000650
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000657
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009377, 0.0009608
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001987, 0.0002100
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008094, 0.0008443
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001269, 0.0001205
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008235, 0.0008418
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020652, 0.0020625
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010040, 0.0009314
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004473, 0.0004864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000645, upper bound: 0.0000650
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000657
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009393, 0.0009592
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001995, 0.0002092
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008118, 0.0008419
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001264, 0.0001209
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008248, 0.0008406
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020650, 0.0020627
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009987, 0.0009366
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004501, 0.0004836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000559, upper bound: 0.0000568
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000559, upper bound: 0.0000568
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009550, 0.0009548
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002079, 0.0002078
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008356, 0.0008353
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001256, 0.0001257
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008373, 0.0008371
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020645, 0.0020645
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009852, 0.0009860
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004784, 0.0004779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000810, upper bound: 0.0000744
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000810, upper bound: 0.0000745
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009555, 0.0009543
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002081, 0.0002076
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008363, 0.0008346
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001255, 0.0001258
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008376, 0.0008367
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020645, 0.0020646
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009837, 0.0009875
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004792, 0.0004771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000683
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000683
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009555, 0.0009542
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002081, 0.0002075
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008364, 0.0008344
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001254, 0.0001258
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008377, 0.0008367
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020644, 0.0020646
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009833, 0.0009876
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004792, 0.0004769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000648, upper bound: 0.0000649
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000648, upper bound: 0.0000649
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009560, 0.0009538
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002084, 0.0002073
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008371, 0.0008337
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001253, 0.0001259
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008380, 0.0008363
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020644, 0.0020646
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009818, 0.0009891
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004800, 0.0004761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000782, upper bound: 0.0000779
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000782, upper bound: 0.0000779
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008885, 0.0008761
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002078, 0.0002017
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007249, 0.0007059
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001160, 0.0001195
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007852, 0.0007754
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020555, 0.0020569
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006908, 0.0007321
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004470, 0.0004250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000794, upper bound: 0.0000756
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000783, upper bound: 0.0000766
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008890, 0.0008756
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002080, 0.0002015
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007256, 0.0007050
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001158, 0.0001197
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007856, 0.0007750
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020554, 0.0020570
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006891, 0.0007336
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004479, 0.0004240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000778, upper bound: 0.0000778
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000775, upper bound: 0.0000817
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008850, 0.0008690
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002034, 0.0001956
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007190, 0.0006945
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001131, 0.0001177
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007824, 0.0007698
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020547, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006615, 0.0007147
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004231, 0.0003947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000675, upper bound: 0.0000636
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000642, upper bound: 0.0000647
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008877, 0.0008642
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002047, 0.0001932
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007232, 0.0006870
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001117, 0.0001185
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007845, 0.0007660
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020541, 0.0020569
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006454, 0.0007238
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004279, 0.0003861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000659, upper bound: 0.0000639
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000641, upper bound: 0.0000656
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008779, 0.0008781
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002000, 0.0002000
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007082, 0.0007084
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001157, 0.0001157
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007768, 0.0007769
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020558, 0.0020557
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006916, 0.0006912
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004105, 0.0004108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000636
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000647
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008805, 0.0008735
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002012, 0.0001978
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007120, 0.0007014
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001144, 0.0001164
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007788, 0.0007734
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020552, 0.0020560
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006766, 0.0006996
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004150, 0.0004027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000663, upper bound: 0.0000639
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000642, upper bound: 0.0000657
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009396, 0.0009602
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001993, 0.0002093
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008123, 0.0008435
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001267, 0.0001210
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008249, 0.0008413
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020652, 0.0020628
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0010015, 0.0009374
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004498, 0.0004841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000645
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000641, upper bound: 0.0000650
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009434, 0.0009554
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002012, 0.0002069
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008182, 0.0008361
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001254, 0.0001222
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008280, 0.0008375
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020646, 0.0020632
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009854, 0.0009502
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004567, 0.0004755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000537
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000538
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008887, 0.0008844
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002063, 0.0002043
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007245, 0.0007180
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001177, 0.0001189
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007853, 0.0007820
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020564, 0.0020569
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007171, 0.0007312
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004425, 0.0004350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000671, upper bound: 0.0000663
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000684
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008912, 0.0008796
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002076, 0.0002019
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007284, 0.0007106
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001163, 0.0001196
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007873, 0.0007782
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020559, 0.0020572
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007010, 0.0007396
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004470, 0.0004264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000658, upper bound: 0.0000677
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000648, upper bound: 0.0000711
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008978, 0.0008933
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002091, 0.0002069
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007420, 0.0007350
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001191, 0.0001203
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007925, 0.0007890
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020574, 0.0020579
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007836, 0.0007987
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004606, 0.0004525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000517
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000517
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008981, 0.0008928
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002092, 0.0002066
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007423, 0.0007342
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001189, 0.0001204
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007927, 0.0007886
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020574, 0.0020580
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007819, 0.0007994
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004609, 0.0004516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000526
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000526
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008633, 0.0008942
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001928, 0.0002091
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006857, 0.0007331
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001205, 0.0001115
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007654, 0.0007897
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020576, 0.0020540
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007498, 0.0006426
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003846, 0.0004522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000542
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000542
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008671, 0.0008898
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001947, 0.0002069
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006915, 0.0007262
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001192, 0.0001126
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007683, 0.0007862
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020571, 0.0020545
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007349, 0.0006552
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003913, 0.0004443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000542
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000542
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008876, 0.0008649
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002044, 0.0001946
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007243, 0.0006887
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001126, 0.0001186
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007845, 0.0007666
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020542, 0.0020569
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006490, 0.0007276
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004284, 0.0003918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000697, upper bound: 0.0000680
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000686
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008742, 0.0008816
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001978, 0.0002039
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007037, 0.0007141
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001175, 0.0001148
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007739, 0.0007797
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020561, 0.0020553
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007085, 0.0006830
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004045, 0.0004339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000536
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000537
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008806, 0.0008733
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002010, 0.0001987
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007134, 0.0007016
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001150, 0.0001166
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007789, 0.0007732
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020552, 0.0020561
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006770, 0.0007042
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004158, 0.0004067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000537
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000541, upper bound: 0.0000538
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008655, 0.0008900
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001936, 0.0002080
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006903, 0.0007271
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001199, 0.0001123
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007671, 0.0007863
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020571, 0.0020543
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007364, 0.0006541
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003891, 0.0004489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000698, upper bound: 0.0000680
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000686
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008921, 0.0008797
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002078, 0.0002031
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007310, 0.0007111
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001170, 0.0001201
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007880, 0.0007782
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020559, 0.0020573
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007020, 0.0007464
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004490, 0.0004307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000574, upper bound: 0.0000564
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000574, upper bound: 0.0000564
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008921, 0.0008797
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002078, 0.0002031
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007310, 0.0007111
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001170, 0.0001201
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007880, 0.0007782
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020559, 0.0020573
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007019, 0.0007465
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004490, 0.0004307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000537
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000537
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008835, 0.0008877
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002035, 0.0002069
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007178, 0.0007235
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001193, 0.0001176
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007812, 0.0007845
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020568, 0.0020563
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007286, 0.0007179
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004338, 0.0004450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000575, upper bound: 0.0000566
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000574, upper bound: 0.0000567
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008830, 0.0008885
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002033, 0.0002073
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007170, 0.0007247
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001195, 0.0001174
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007808, 0.0007851
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020569, 0.0020563
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007313, 0.0007162
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004329, 0.0004461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000709, upper bound: 0.0000696
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000703, upper bound: 0.0000698
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009644, 0.0009552
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002103, 0.0002069
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008502, 0.0008356
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001250, 0.0001275
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008446, 0.0008374
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020646, 0.0020656
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009851, 0.0010170
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004891, 0.0004747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000683, upper bound: 0.0000655
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000656
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009644, 0.0009552
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002103, 0.0002068
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008503, 0.0008355
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001250, 0.0001275
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008447, 0.0008374
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020646, 0.0020656
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009850, 0.0010171
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004892, 0.0004747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000682, upper bound: 0.0000660
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000683, upper bound: 0.0000662
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009609, 0.0009540
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002097, 0.0002073
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008454, 0.0008341
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001254, 0.0001272
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008419, 0.0008365
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020644, 0.0020652
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009826, 0.0010071
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004874, 0.0004765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000674
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000701, upper bound: 0.0000678
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009626, 0.0009524
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002105, 0.0002066
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008480, 0.0008317
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001249, 0.0001277
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008432, 0.0008352
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020642, 0.0020654
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009774, 0.0010126
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004903, 0.0004737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000674
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000698, upper bound: 0.0000678
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008919, 0.0008748
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002089, 0.0002018
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007309, 0.0007039
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001161, 0.0001204
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007879, 0.0007744
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020553, 0.0020573
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006872, 0.0007471
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004523, 0.0004259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000698
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000698
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008935, 0.0008732
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002096, 0.0002010
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007333, 0.0007015
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001156, 0.0001209
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007891, 0.0007732
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020551, 0.0020575
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006820, 0.0007522
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004551, 0.0004231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000687
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000687
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008835, 0.0008835
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002047, 0.0002060
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007179, 0.0007172
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001185, 0.0001180
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007812, 0.0007812
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020563, 0.0020563
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007161, 0.0007189
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004373, 0.0004413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000700
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000700
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008851, 0.0008820
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002055, 0.0002053
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007204, 0.0007149
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001181, 0.0001185
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007825, 0.0007800
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020561, 0.0020565
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007110, 0.0007244
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004402, 0.0004386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000702, upper bound: 0.0000705
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000702, upper bound: 0.0000705
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009607, 0.0009434
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002084, 0.0002010
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008450, 0.0008177
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001218, 0.0001266
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008416, 0.0008279
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020632, 0.0020652
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009494, 0.0010086
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004844, 0.0004554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000642
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000641
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009633, 0.0009395
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002097, 0.0001991
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008490, 0.0008118
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001207, 0.0001274
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008437, 0.0008249
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020627, 0.0020655
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009366, 0.0010174
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004891, 0.0004485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000643
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000642
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009115, 0.0008830
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002146, 0.0002018
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007634, 0.0007191
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001161, 0.0001240
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008032, 0.0007808
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020562, 0.0020595
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007491, 0.0008476
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004828, 0.0004340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000655, upper bound: 0.0000644
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000648
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009043, 0.0008917
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002111, 0.0002061
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007524, 0.0007325
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001186, 0.0001219
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007976, 0.0007877
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020572, 0.0020587
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007781, 0.0008237
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004700, 0.0004496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000654, upper bound: 0.0000643
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000647
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008893, 0.0008639
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002050, 0.0001940
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007265, 0.0006869
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001121, 0.0001188
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007858, 0.0007658
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020541, 0.0020571
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006448, 0.0007316
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004287, 0.0003878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000694
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000694
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008821, 0.0008724
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002015, 0.0001981
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007154, 0.0006999
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001145, 0.0001168
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007801, 0.0007725
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020551, 0.0020562
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006730, 0.0007076
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004159, 0.0004029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000639, upper bound: 0.0000659
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000635, upper bound: 0.0000659
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009637, 0.0009391
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002099, 0.0001989
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008495, 0.0008111
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001206, 0.0001275
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008439, 0.0008246
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020627, 0.0020656
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009352, 0.0010185
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004897, 0.0004478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000636, upper bound: 0.0000676
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000636, upper bound: 0.0000675
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009637, 0.0009391
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002099, 0.0001989
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008496, 0.0008111
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001206, 0.0001275
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008440, 0.0008245
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020627, 0.0020656
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009351, 0.0010187
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004898, 0.0004477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000657
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000657
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008982, 0.0008999
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002082, 0.0002106
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007430, 0.0007439
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001207, 0.0001202
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007928, 0.0007942
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020581, 0.0020580
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008040, 0.0008033
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004591, 0.0004663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000523, upper bound: 0.0000515
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000523, upper bound: 0.0000515
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008982, 0.0008999
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002082, 0.0002106
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007431, 0.0007439
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001207, 0.0001202
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007928, 0.0007942
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020581, 0.0020580
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008039, 0.0008034
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004592, 0.0004662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000522, upper bound: 0.0000517
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000522, upper bound: 0.0000517
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008856, 0.0009062
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002030, 0.0002146
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007240, 0.0007538
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001232, 0.0001172
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007828, 0.0007991
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020588, 0.0020565
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008231, 0.0007602
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004401, 0.0004809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000522, upper bound: 0.0000514
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000517
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008872, 0.0009046
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002037, 0.0002139
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007264, 0.0007515
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001227, 0.0001177
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007841, 0.0007979
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020587, 0.0020567
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008180, 0.0007654
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004429, 0.0004782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000684
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000687
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008947, 0.0008971
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002074, 0.0002102
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007380, 0.0007399
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001206, 0.0001198
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007900, 0.0007920
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020578, 0.0020576
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007929, 0.0007906
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004564, 0.0004648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000694
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000680, upper bound: 0.0000697
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008963, 0.0008955
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002082, 0.0002094
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007404, 0.0007375
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001201, 0.0001203
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007912, 0.0007907
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020576, 0.0020578
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007876, 0.0007957
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004591, 0.0004620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000517, upper bound: 0.0000526
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000526
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008860, 0.0009057
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002032, 0.0002144
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007247, 0.0007532
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001231, 0.0001173
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007832, 0.0007987
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020588, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008216, 0.0007616
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004409, 0.0004802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000693
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000680, upper bound: 0.0000696
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008876, 0.0009042
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002039, 0.0002137
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007270, 0.0007508
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001226, 0.0001178
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007844, 0.0007976
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020586, 0.0020568
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0008166, 0.0007668
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004437, 0.0004775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000517, upper bound: 0.0000526
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000526
time: 0.63 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000634
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000642
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000634
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000642
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000677, upper bound: 0.0000634
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000641
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000677, upper bound: 0.0000634
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000641
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000645, upper bound: 0.0000650
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000657
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000645, upper bound: 0.0000650
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000657
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000645, upper bound: 0.0000650
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000657
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000559, upper bound: 0.0000568
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000559, upper bound: 0.0000568
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000810, upper bound: 0.0000744
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000810, upper bound: 0.0000745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000683
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000683
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000648, upper bound: 0.0000649
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000648, upper bound: 0.0000649
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000782, upper bound: 0.0000779
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000782, upper bound: 0.0000779
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000794, upper bound: 0.0000756
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000783, upper bound: 0.0000766
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000778, upper bound: 0.0000778
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000775, upper bound: 0.0000817
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000675, upper bound: 0.0000636
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000642, upper bound: 0.0000647
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000659, upper bound: 0.0000639
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000641, upper bound: 0.0000656
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000676, upper bound: 0.0000636
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000647
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000663, upper bound: 0.0000639
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000642, upper bound: 0.0000657
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000645
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000641, upper bound: 0.0000650
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000537
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000538
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000671, upper bound: 0.0000663
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000684
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000658, upper bound: 0.0000677
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000648, upper bound: 0.0000711
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000517
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000517
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000526
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000526
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000542
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000542
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000542
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000542
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000697, upper bound: 0.0000680
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000686
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000536
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000537
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000537
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000541, upper bound: 0.0000538
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000698, upper bound: 0.0000680
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000686
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000574, upper bound: 0.0000564
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000574, upper bound: 0.0000564
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000537
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000542, upper bound: 0.0000537
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000575, upper bound: 0.0000566
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000574, upper bound: 0.0000567
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000709, upper bound: 0.0000696
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000703, upper bound: 0.0000698
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000683, upper bound: 0.0000655
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000656
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000682, upper bound: 0.0000660
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000683, upper bound: 0.0000662
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000674
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000701, upper bound: 0.0000678
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000700, upper bound: 0.0000674
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000698, upper bound: 0.0000678
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000698
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000698
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000687
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000687
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000700
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000704, upper bound: 0.0000700
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000702, upper bound: 0.0000705
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000702, upper bound: 0.0000705
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000657, upper bound: 0.0000642
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000656, upper bound: 0.0000641
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000643
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000642
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000655, upper bound: 0.0000644
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000648
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000654, upper bound: 0.0000643
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000647, upper bound: 0.0000647
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000694
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000694
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000639, upper bound: 0.0000659
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000635, upper bound: 0.0000659
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000636, upper bound: 0.0000676
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000636, upper bound: 0.0000675
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000657
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000643, upper bound: 0.0000657
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000523, upper bound: 0.0000515
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000523, upper bound: 0.0000515
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000522, upper bound: 0.0000517
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000522, upper bound: 0.0000517
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000522, upper bound: 0.0000514
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000517
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000687, upper bound: 0.0000684
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000686, upper bound: 0.0000687
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000680, upper bound: 0.0000697
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000517, upper bound: 0.0000526
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000526
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000684, upper bound: 0.0000693
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000680, upper bound: 0.0000696
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000517, upper bound: 0.0000526
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000526

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008819, 0.0008666
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002029, 0.0001954
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007145, 0.0006909
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001129, 0.0001173
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007800, 0.0007679
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020544, 0.0020562
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006535, 0.0007046
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004201, 0.0003928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008823, 0.0008662
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002031, 0.0001952
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007150, 0.0006902
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001128, 0.0001174
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007803, 0.0007676
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020544, 0.0020563
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006520, 0.0007057
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004207, 0.0003920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008686, 0.0008833
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001963, 0.0002049
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006939, 0.0007164
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001178, 0.0001134
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007695, 0.0007811
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020563, 0.0020547
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007145, 0.0006600
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003963, 0.0004371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008690, 0.0008828
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001966, 0.0002047
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006946, 0.0007157
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001177, 0.0001136
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007698, 0.0007807
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020562, 0.0020547
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007130, 0.0006616
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003971, 0.0004363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008749, 0.0008756
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001994, 0.0001998
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007036, 0.0007047
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001155, 0.0001152
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007744, 0.0007750
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020555, 0.0020554
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006835, 0.0006810
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004075, 0.0004088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008751, 0.0008751
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001995, 0.0001996
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007039, 0.0007040
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001153, 0.0001153
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007746, 0.0007746
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020554, 0.0020554
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006819, 0.0006818
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004079, 0.0004080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008599, 0.0008923
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001921, 0.0002093
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006805, 0.0007303
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001204, 0.0001109
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007626, 0.0007882
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020574, 0.0020536
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007446, 0.0006310
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003808, 0.0004531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008603, 0.0008918
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001923, 0.0002091
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006812, 0.0007295
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001202, 0.0001111
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007630, 0.0007878
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020573, 0.0020537
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007430, 0.0006325
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003816, 0.0004523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009504, 0.0009429
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002056, 0.0002019
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008289, 0.0008174
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001224, 0.0001245
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008335, 0.0008276
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020631, 0.0020640
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009496, 0.0009745
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004720, 0.0004587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000519
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000520
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009506, 0.0009425
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002057, 0.0002017
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008293, 0.0008167
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001223, 0.0001246
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008337, 0.0008273
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020631, 0.0020640
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009481, 0.0009752
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004724, 0.0004579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000524
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009515, 0.0009414
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002061, 0.0002011
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008307, 0.0008150
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001219, 0.0001248
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008344, 0.0008264
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020629, 0.0020641
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009443, 0.0009783
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004741, 0.0004559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000519
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000520
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009518, 0.0009409
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002063, 0.0002009
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008311, 0.0008143
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001218, 0.0001249
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008346, 0.0008260
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020629, 0.0020642
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009428, 0.0009793
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004746, 0.0004551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000524
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009356, 0.0009590
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001983, 0.0002098
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008061, 0.0008417
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001268, 0.0001203
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008218, 0.0008404
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020650, 0.0020623
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009992, 0.0009251
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004457, 0.0004854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000519
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000520
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009361, 0.0009585
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001986, 0.0002096
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008069, 0.0008410
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001267, 0.0001204
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008222, 0.0008400
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020649, 0.0020623
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009977, 0.0009267
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004465, 0.0004846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000524
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008848, 0.0008761
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002068, 0.0002025
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007191, 0.0007057
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001164, 0.0001189
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007823, 0.0007754
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020555, 0.0020565
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006913, 0.0007204
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004439, 0.0004283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000508
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000508
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008763, 0.0008852
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002026, 0.0002070
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007061, 0.0007198
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001191, 0.0001165
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007756, 0.0007826
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020565, 0.0020555
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007218, 0.0006921
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004288, 0.0004446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000509
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000509
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009546, 0.0009535
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002077, 0.0002072
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008349, 0.0008333
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001252, 0.0001255
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008369, 0.0008361
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020644, 0.0020645
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009810, 0.0009844
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004776, 0.0004758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000554, upper bound: 0.0000550
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000550
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009547, 0.0009543
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002077, 0.0002076
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008351, 0.0008346
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001255, 0.0001255
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008370, 0.0008367
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020645, 0.0020645
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009837, 0.0009848
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004778, 0.0004771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000554, upper bound: 0.0000550
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000550
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009531, 0.0009382
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002069, 0.0001996
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008330, 0.0008101
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001210, 0.0001253
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008356, 0.0008239
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020626, 0.0020643
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009337, 0.0009834
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004768, 0.0004502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000519
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000520
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009395, 0.0009542
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002002, 0.0002075
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008121, 0.0008344
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001254, 0.0001214
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008249, 0.0008367
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020644, 0.0020627
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009833, 0.0009380
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004525, 0.0004769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000519
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000520
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008857, 0.0008750
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002072, 0.0002020
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007206, 0.0007041
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001161, 0.0001192
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007830, 0.0007746
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020553, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006879, 0.0007235
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004455, 0.0004265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008772, 0.0008842
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002031, 0.0002065
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007075, 0.0007183
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001188, 0.0001168
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007763, 0.0007818
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020564, 0.0020556
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007185, 0.0006952
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004304, 0.0004429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008863, 0.0008744
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002075, 0.0002017
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007215, 0.0007032
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001160, 0.0001194
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007835, 0.0007741
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020553, 0.0020567
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006858, 0.0007255
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004466, 0.0004254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000543
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000543
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008868, 0.0008739
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002078, 0.0002015
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007222, 0.0007025
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001158, 0.0001195
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007838, 0.0007737
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020552, 0.0020567
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006843, 0.0007271
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004475, 0.0004246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000554, upper bound: 0.0000551
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000554, upper bound: 0.0000551
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008868, 0.0008739
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002078, 0.0002014
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007222, 0.0007023
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001158, 0.0001195
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007838, 0.0007737
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020552, 0.0020567
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006840, 0.0007271
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004475, 0.0004244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008873, 0.0008734
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002080, 0.0002012
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007230, 0.0007017
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001157, 0.0001196
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007842, 0.0007733
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020552, 0.0020568
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006825, 0.0007287
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004483, 0.0004236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000546, upper bound: 0.0000567
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000546, upper bound: 0.0000567
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008820, 0.0008666
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002029, 0.0001954
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007145, 0.0006909
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001129, 0.0001173
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007800, 0.0007679
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020544, 0.0020562
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006535, 0.0007047
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004201, 0.0003928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008822, 0.0008660
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002030, 0.0001951
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007149, 0.0006900
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001127, 0.0001173
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007802, 0.0007675
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020544, 0.0020563
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006515, 0.0007055
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004206, 0.0003917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000520
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000520
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008847, 0.0008617
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002042, 0.0001930
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007187, 0.0006834
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001115, 0.0001181
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007822, 0.0007641
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020539, 0.0020565
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006373, 0.0007138
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004250, 0.0003841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000514
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000513
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008850, 0.0008612
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002044, 0.0001927
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007192, 0.0006825
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001113, 0.0001181
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007824, 0.0007637
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020538, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006354, 0.0007148
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004255, 0.0003831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000520
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000520
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008749, 0.0008756
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001995, 0.0001998
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007037, 0.0007047
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001155, 0.0001153
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007745, 0.0007750
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020555, 0.0020554
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006835, 0.0006812
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004076, 0.0004088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008751, 0.0008751
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001996, 0.0001995
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007040, 0.0007039
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001153, 0.0001153
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007746, 0.0007746
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020554, 0.0020554
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006817, 0.0006819
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004080, 0.0004078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000521
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000521
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008775, 0.0008711
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002007, 0.0001976
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007076, 0.0006977
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001142, 0.0001160
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007765, 0.0007714
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020549, 0.0020557
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006684, 0.0006897
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004121, 0.0004007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000514
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000514
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008778, 0.0008705
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002009, 0.0001973
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007081, 0.0006969
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001140, 0.0001161
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007767, 0.0007710
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020549, 0.0020557
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006666, 0.0006908
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004127, 0.0003998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000521
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000521
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009368, 0.0009580
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001989, 0.0002093
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008080, 0.0008402
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001265, 0.0001206
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008228, 0.0008396
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020649, 0.0020624
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009959, 0.0009291
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004478, 0.0004836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009372, 0.0009574
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001991, 0.0002091
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008086, 0.0008393
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001263, 0.0001207
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008231, 0.0008392
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020648, 0.0020625
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009939, 0.0009305
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004485, 0.0004826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000520
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000521
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008857, 0.0008820
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002061, 0.0002043
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007201, 0.0007145
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001174, 0.0001185
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007830, 0.0007801
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020561, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007102, 0.0007224
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004414, 0.0004349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000554, upper bound: 0.0000553
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000552, upper bound: 0.0000553
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008861, 0.0008815
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002063, 0.0002040
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007207, 0.0007136
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001173, 0.0001186
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007833, 0.0007796
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020561, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007083, 0.0007237
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004421, 0.0004339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000544, upper bound: 0.0000563
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000544, upper bound: 0.0000565
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008882, 0.0008772
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002073, 0.0002019
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007240, 0.0007070
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001160, 0.0001192
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007849, 0.0007763
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020556, 0.0020569
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006940, 0.0007308
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004459, 0.0004263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000552, upper bound: 0.0000553
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008886, 0.0008766
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002075, 0.0002017
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007246, 0.0007062
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001159, 0.0001193
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007853, 0.0007758
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020555, 0.0020569
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006922, 0.0007322
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004467, 0.0004253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000544, upper bound: 0.0000563
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000544, upper bound: 0.0000568
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008846, 0.0008625
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002039, 0.0001944
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007198, 0.0006851
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001124, 0.0001182
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007821, 0.0007647
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020540, 0.0020565
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006409, 0.0007175
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004248, 0.0003894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008849, 0.0008620
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002040, 0.0001941
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007202, 0.0006843
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001123, 0.0001183
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007823, 0.0007643
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020539, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006392, 0.0007183
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004252, 0.0003885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000519
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000520
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008626, 0.0008875
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001931, 0.0002080
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006859, 0.0007234
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001197, 0.0001119
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007647, 0.0007844
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020568, 0.0020539
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007296, 0.0006439
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003855, 0.0004485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008630, 0.0008871
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001933, 0.0002078
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006865, 0.0007227
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001196, 0.0001120
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007651, 0.0007840
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020567, 0.0020540
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007280, 0.0006453
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003862, 0.0004477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000520
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000521
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008800, 0.0008860
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002030, 0.0002072
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007126, 0.0007210
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001193, 0.0001170
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007785, 0.0007832
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020566, 0.0020559
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007244, 0.0007072
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004312, 0.0004457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000544
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000546
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008804, 0.0008855
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002032, 0.0002070
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007132, 0.0007203
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001191, 0.0001171
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007788, 0.0007828
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020566, 0.0020560
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007229, 0.0007085
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004319, 0.0004449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000550, upper bound: 0.0000555
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008942, 0.0008767
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002089, 0.0002017
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007339, 0.0007063
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001159, 0.0001204
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007897, 0.0007759
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020555, 0.0020576
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006924, 0.0007531
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004523, 0.0004254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000564, upper bound: 0.0000543
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000543
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008859, 0.0008854
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002048, 0.0002059
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007210, 0.0007196
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001184, 0.0001180
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007831, 0.0007827
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020565, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007214, 0.0007253
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004374, 0.0004409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000565, upper bound: 0.0000544
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000544
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008943, 0.0008766
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002089, 0.0002017
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007339, 0.0007062
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001159, 0.0001204
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007897, 0.0007759
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020555, 0.0020576
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006923, 0.0007531
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004523, 0.0004253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000546
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000561, upper bound: 0.0000546
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008859, 0.0008854
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002049, 0.0002059
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007211, 0.0007196
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001184, 0.0001180
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007832, 0.0007827
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020565, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007213, 0.0007254
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004375, 0.0004408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000546
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000546
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009606, 0.0009537
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002096, 0.0002073
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008449, 0.0008336
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001253, 0.0001271
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008417, 0.0008363
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020644, 0.0020652
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009817, 0.0010061
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004868, 0.0004760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000564, upper bound: 0.0000543
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000565, upper bound: 0.0000544
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009607, 0.0009537
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002096, 0.0002073
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008449, 0.0008336
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001253, 0.0001271
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008417, 0.0008362
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020644, 0.0020652
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009816, 0.0010062
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004868, 0.0004760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000546
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000546
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009623, 0.0009522
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002104, 0.0002065
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008474, 0.0008312
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001248, 0.0001276
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008430, 0.0008350
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020642, 0.0020654
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009765, 0.0010116
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004897, 0.0004733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000543
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000544
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009623, 0.0009521
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002104, 0.0002065
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008475, 0.0008312
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001248, 0.0001276
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008430, 0.0008350
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020642, 0.0020654
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009764, 0.0010117
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004898, 0.0004732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000561, upper bound: 0.0000546
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000546
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008909, 0.0008740
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002083, 0.0002014
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007292, 0.0007027
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001158, 0.0001201
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007870, 0.0007738
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020552, 0.0020572
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006845, 0.0007433
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004505, 0.0004246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000551
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000553
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008912, 0.0008748
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002085, 0.0002018
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007297, 0.0007039
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001161, 0.0001202
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007873, 0.0007744
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020553, 0.0020572
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006872, 0.0007443
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004510, 0.0004259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000551
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000553
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008889, 0.0008566
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002060, 0.0001915
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007264, 0.0006760
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001107, 0.0001194
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007855, 0.0007600
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020533, 0.0020570
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006212, 0.0007318
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004324, 0.0003789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000519
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000520
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008768, 0.0008732
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002001, 0.0002010
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007078, 0.0007015
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001156, 0.0001160
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007760, 0.0007732
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020551, 0.0020556
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006820, 0.0006914
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004109, 0.0004231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000519
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000520
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008826, 0.0008827
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002043, 0.0002056
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007165, 0.0007160
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001183, 0.0001178
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007805, 0.0007806
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020562, 0.0020562
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007133, 0.0007157
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004357, 0.0004400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000554
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008827, 0.0008835
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002043, 0.0002060
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007167, 0.0007172
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001185, 0.0001178
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007806, 0.0007812
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020563, 0.0020562
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007161, 0.0007162
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004360, 0.0004413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000554
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008842, 0.0008812
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002051, 0.0002049
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007190, 0.0007137
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001179, 0.0001182
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007818, 0.0007794
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020561, 0.0020564
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007083, 0.0007211
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004386, 0.0004373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000554
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000555
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008844, 0.0008820
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002051, 0.0002053
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007192, 0.0007149
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001181, 0.0001183
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007819, 0.0007800
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020561, 0.0020564
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007110, 0.0007217
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004389, 0.0004386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000554
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000555
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008888, 0.0008642
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002049, 0.0001942
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007256, 0.0006871
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001122, 0.0001187
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007854, 0.0007660
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020541, 0.0020570
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006453, 0.0007295
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004279, 0.0003884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008816, 0.0008726
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002014, 0.0001983
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007145, 0.0007001
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001146, 0.0001166
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007797, 0.0007727
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020551, 0.0020562
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006735, 0.0007056
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004151, 0.0004035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008915, 0.0008603
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002062, 0.0001923
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007297, 0.0006812
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001111, 0.0001194
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007875, 0.0007630
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020537, 0.0020573
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006325, 0.0007385
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004327, 0.0003816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000510
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000510
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008842, 0.0008690
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002027, 0.0001966
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007186, 0.0006946
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001136, 0.0001173
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007818, 0.0007698
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020547, 0.0020565
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006615, 0.0007143
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004197, 0.0003971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000510
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000510
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008888, 0.0008641
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002049, 0.0001942
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007256, 0.0006871
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001122, 0.0001187
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007854, 0.0007660
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020541, 0.0020570
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006453, 0.0007296
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004279, 0.0003884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000515
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000515
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008915, 0.0008603
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002062, 0.0001923
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007298, 0.0006811
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001111, 0.0001194
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007875, 0.0007629
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020537, 0.0020573
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006324, 0.0007387
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004328, 0.0003815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000514
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000514
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008816, 0.0008726
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002014, 0.0001983
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007146, 0.0007001
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001146, 0.0001166
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007798, 0.0007726
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020551, 0.0020562
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006734, 0.0007058
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004152, 0.0004034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000516
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000515
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008843, 0.0008690
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002027, 0.0001966
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007186, 0.0006945
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001136, 0.0001174
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007818, 0.0007698
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020547, 0.0020565
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006614, 0.0007145
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004198, 0.0003970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000515
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000515
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008852, 0.0008616
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002042, 0.0001939
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007207, 0.0006836
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001121, 0.0001184
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007826, 0.0007639
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020538, 0.0020566
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006378, 0.0007194
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004259, 0.0003877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000509, upper bound: 0.0000524
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008864, 0.0008599
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002048, 0.0001931
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007225, 0.0006811
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001117, 0.0001187
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007835, 0.0007626
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020537, 0.0020567
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006323, 0.0007234
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004280, 0.0003848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000509, upper bound: 0.0000524
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008818, 0.0008721
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002015, 0.0001981
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007148, 0.0006994
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001145, 0.0001167
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007799, 0.0007723
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020551, 0.0020562
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006719, 0.0007063
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004154, 0.0004026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008818, 0.0008721
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002015, 0.0001981
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007149, 0.0006994
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001145, 0.0001167
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007799, 0.0007723
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020551, 0.0020562
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006719, 0.0007064
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004155, 0.0004026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000508, upper bound: 0.0000524
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000508, upper bound: 0.0000524
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008918, 0.0008599
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002064, 0.0001921
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007302, 0.0006806
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001110, 0.0001195
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007877, 0.0007627
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020536, 0.0020574
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006311, 0.0007396
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004332, 0.0003808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008845, 0.0008686
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002028, 0.0001964
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007191, 0.0006939
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001134, 0.0001174
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007821, 0.0007695
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020547, 0.0020565
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006601, 0.0007155
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004204, 0.0003963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009598, 0.0009368
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002092, 0.0001989
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008441, 0.0008079
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001206, 0.0001271
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008409, 0.0008228
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020624, 0.0020651
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009290, 0.0010074
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004872, 0.0004478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0009612, 0.0009352
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0002098, 0.0001981
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0008462, 0.0008055
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001202, 0.0001275
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0008420, 0.0008215
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020622, 0.0020653
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0009238, 0.0010119
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004896, 0.0004449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008646, 0.0008855
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001941, 0.0002070
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006890, 0.0007203
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001191, 0.0001125
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007663, 0.0007828
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020566, 0.0020542
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007228, 0.0006507
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003891, 0.0004449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000516
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008684, 0.0008819
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001960, 0.0002053
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006949, 0.0007148
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001181, 0.0001136
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007694, 0.0007800
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020561, 0.0020546
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007109, 0.0006635
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003960, 0.0004385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000510
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000515
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008721, 0.0008782
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001978, 0.0002035
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007006, 0.0007091
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001170, 0.0001146
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007723, 0.0007771
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020557, 0.0020551
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006986, 0.0006759
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004026, 0.0004319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000509, upper bound: 0.0000524
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008757, 0.0008744
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001996, 0.0002016
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0007061, 0.0007032
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001159, 0.0001157
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007751, 0.0007741
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020553, 0.0020555
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0006858, 0.0006878
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0004090, 0.0004251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008634, 0.0008866
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001936, 0.0002075
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006872, 0.0007220
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001194, 0.0001121
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007654, 0.0007837
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020567, 0.0020541
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007265, 0.0006469
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003871, 0.0004468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000508, upper bound: 0.0000524
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0000911, 0.0012060, -0.0000911, 0.0012060, -0.0008673, 0.0008830
1: -0.0034967, -0.0031792, -0.0034967, -0.0031792, -0.0001954, 0.0002058
2: 0.0148218, 0.0161782, 0.0148218, 0.0161782, -0.0006931, 0.0007165
3: 1.0066561, 1.0069827, 1.0066561, 1.0069827, -0.0003266, 0.0003266
4: -0.0042765, -0.0040696, -0.0042765, -0.0040696, -0.0001184, 0.0001132
5: 0.0039104, 0.0050175, 0.0039104, 0.0050175, -0.0007684, 0.0007809
6: -0.0028078, -0.0025578, -0.0028078, -0.0025578, -0.0002500, 0.0002500
7: -0.0133424, -0.0112397, -0.0133424, -0.0112397, -0.0020563, 0.0020545
8: -0.0139705, -0.0118393, -0.0139705, -0.0118393, -0.0007145, 0.0006596
9: 0.0017492, 0.0027556, 0.0017492, 0.0027556, -0.0003939, 0.0004404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
time: 0.75 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000519
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000520
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000524
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000519
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000520
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000524
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000519
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000520
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000524
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000508
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000508
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000509
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000509
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000554, upper bound: 0.0000550
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000550
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000554, upper bound: 0.0000550
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000550
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000519
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000520
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000519
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000520
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000543
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000543
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000554, upper bound: 0.0000551
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000554, upper bound: 0.0000551
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000546, upper bound: 0.0000567
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000546, upper bound: 0.0000567
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000520
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000520
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000514
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000513
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000520
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000520
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000521
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000521
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000514
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000514
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000521
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000521
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000520
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000521
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000554, upper bound: 0.0000553
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000552, upper bound: 0.0000553
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000544, upper bound: 0.0000563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000544, upper bound: 0.0000565
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000552, upper bound: 0.0000553
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000544, upper bound: 0.0000563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000544, upper bound: 0.0000568
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000519
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000520
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000511
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000515
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000520
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000510, upper bound: 0.0000521
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000544
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000546
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000550, upper bound: 0.0000555
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000564, upper bound: 0.0000543
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000543
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000565, upper bound: 0.0000544
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000544
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000561, upper bound: 0.0000546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000564, upper bound: 0.0000543
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000565, upper bound: 0.0000544
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000543
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000563, upper bound: 0.0000544
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000561, upper bound: 0.0000546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000546
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000551
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000553
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000551
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000553
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000519
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000520
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000516, upper bound: 0.0000519
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000513, upper bound: 0.0000520
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000553
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000555
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000553, upper bound: 0.0000554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000551, upper bound: 0.0000555
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000513
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000521, upper bound: 0.0000510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000515
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000515
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000514
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000514
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000516
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000515
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000515
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000515
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000509, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000509, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000508, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000508, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000513
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000516
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000520, upper bound: 0.0000510
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000515
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000509, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000514, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000508, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000515, upper bound: 0.0000524
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.56
Output dim: 3, lower bound: -0.0000511, upper bound: 0.0000524

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.06 + 581.80 = 584.86 seconds
