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
0: (-0.0002960, 0.0001355, -0.0002960, 0.0001355, -0.0003539, 0.0003539)
1: (-0.0000653, 0.0015293, -0.0000653, 0.0015293, -0.0013536, 0.0013536)
2: (0.0140497, 0.0164378, 0.0140497, 0.0164378, -0.0020067, 0.0020067)
3: (-0.0000621, 0.0017337, -0.0000621, 0.0017337, -0.0015001, 0.0015001)
4: (-0.0044369, -0.0027805, -0.0044369, -0.0027805, -0.0014684, 0.0014684)
5: (0.0078762, 0.0096687, 0.0078762, 0.0096687, -0.0014965, 0.0014965)
6: (0.0092842, 0.0099607, 0.0092842, 0.0099607, -0.0006764, 0.0006764)
7: (-0.0193891, -0.0154978, -0.0193891, -0.0154978, -0.0031567, 0.0031567)
8: (0.9682389, 0.9793879, 0.9682389, 0.9793879, -0.0093888, 0.0093888)
9: (0.0037110, 0.0069878, 0.0037110, 0.0069878, -0.0026880, 0.0026880)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.57 = 2.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0065366, upper bound: 0.0065366

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060068, upper bound: 0.0061654
time: 0.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061654, upper bound: 0.0061654
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.30 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.30
Output dim: 8, lower bound: -0.0060068, upper bound: 0.0061654
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.30
Output dim: 8, lower bound: -0.0061654, upper bound: 0.0061654

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0002703, 0.0001218, -0.0002931, 0.0001353, -0.0003230, 0.0003255
1: 0.0000547, 0.0015084, -0.0000518, 0.0015289, -0.0012144, 0.0012662
2: 0.0140810, 0.0162581, 0.0140502, 0.0164175, -0.0018745, 0.0017957
3: -0.0000385, 0.0015985, -0.0000617, 0.0017184, -0.0014004, 0.0013404
4: -0.0044152, -0.0029052, -0.0044366, -0.0027946, -0.0013793, 0.0013241
5: 0.0078997, 0.0095337, 0.0078766, 0.0096534, -0.0013970, 0.0013370
6: 0.0093351, 0.0099518, 0.0092900, 0.0099605, -0.0006254, 0.0006618
7: -0.0190962, -0.0155488, -0.0193560, -0.0154986, -0.0028045, 0.0029406
8: 0.9690781, 0.9792416, 0.9683338, 0.9793854, -0.0084067, 0.0087726
9: 0.0037540, 0.0067411, 0.0037118, 0.0069599, -0.0025054, 0.0023909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060061, upper bound: 0.0060062
time: 0.68 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060061, upper bound: 0.0061654
time: 0.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0002832, 0.0001344, -0.0002960, 0.0001355, -0.0003190, 0.0003527
1: -0.0000054, 0.0015277, -0.0000653, 0.0015293, -0.0012183, 0.0013522
2: 0.0140522, 0.0163480, 0.0140497, 0.0164378, -0.0020046, 0.0017959
3: -0.0000603, 0.0016661, -0.0000621, 0.0017337, -0.0014985, 0.0013378
4: -0.0044352, -0.0028428, -0.0044369, -0.0027805, -0.0014670, 0.0013386
5: 0.0078780, 0.0096012, 0.0078762, 0.0096687, -0.0014950, 0.0013342
6: 0.0093097, 0.0099600, 0.0092842, 0.0099607, -0.0006510, 0.0006757
7: -0.0192427, -0.0155018, -0.0193891, -0.0154978, -0.0027741, 0.0031533
8: 0.9686583, 0.9793764, 0.9682389, 0.9793879, -0.0084137, 0.0093791
9: 0.0037144, 0.0068645, 0.0037110, 0.0069878, -0.0026852, 0.0023731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061654, upper bound: 0.0060062
time: 0.67 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061654, upper bound: 0.0061654
time: 0.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.62 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 8, lower bound: -0.0060061, upper bound: 0.0060062
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 8, lower bound: -0.0060061, upper bound: 0.0061654
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 8, lower bound: -0.0061654, upper bound: 0.0060062
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 8, lower bound: -0.0061654, upper bound: 0.0061654

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002703, 0.0001218, -0.0002703, 0.0001218, -0.0002986, 0.0002986
1: 0.0000547, 0.0015084, 0.0000547, 0.0015084, -0.0011442, 0.0011442
2: 0.0140810, 0.0162581, 0.0140810, 0.0162581, -0.0016905, 0.0016905
3: -0.0000385, 0.0015985, -0.0000385, 0.0015985, -0.0012613, 0.0012613
4: -0.0044152, -0.0029052, -0.0044152, -0.0029052, -0.0012511, 0.0012511
5: 0.0078997, 0.0095337, 0.0078997, 0.0095337, -0.0012580, 0.0012580
6: 0.0093351, 0.0099518, 0.0093351, 0.0099518, -0.0006166, 0.0006166
7: -0.0190962, -0.0155488, -0.0190962, -0.0155488, -0.0026331, 0.0026331
8: 0.9690781, 0.9792416, 0.9690781, 0.9792416, -0.0079156, 0.0079156
9: 0.0037540, 0.0067411, 0.0037540, 0.0067411, -0.0022466, 0.0022466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057777, upper bound: 0.0055988
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058348, upper bound: 0.0058335
time: 0.66 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002703, 0.0001218, -0.0002832, 0.0001344, -0.0003221, 0.0003214
1: 0.0000547, 0.0015084, -0.0000054, 0.0015277, -0.0012133, 0.0012415
2: 0.0140810, 0.0162581, 0.0140522, 0.0163480, -0.0018414, 0.0017940
3: -0.0000385, 0.0015985, -0.0000603, 0.0016661, -0.0013765, 0.0013391
4: -0.0044152, -0.0029052, -0.0044352, -0.0028428, -0.0013351, 0.0013229
5: 0.0078997, 0.0095337, 0.0078780, 0.0096012, -0.0013733, 0.0013358
6: 0.0093351, 0.0099518, 0.0093097, 0.0099600, -0.0006248, 0.0006421
7: -0.0190962, -0.0155488, -0.0192427, -0.0155018, -0.0028018, 0.0028940
8: 0.9690781, 0.9792416, 0.9686583, 0.9793764, -0.0083990, 0.0086154
9: 0.0037540, 0.0067411, 0.0037144, 0.0068645, -0.0024644, 0.0023887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057777, upper bound: 0.0057761
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058348, upper bound: 0.0059911
time: 0.73 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002832, 0.0001344, -0.0002703, 0.0001218, -0.0003214, 0.0003221
1: -0.0000054, 0.0015277, 0.0000547, 0.0015084, -0.0012415, 0.0012133
2: 0.0140522, 0.0163480, 0.0140810, 0.0162581, -0.0017940, 0.0018414
3: -0.0000603, 0.0016661, -0.0000385, 0.0015985, -0.0013391, 0.0013765
4: -0.0044352, -0.0028428, -0.0044152, -0.0029052, -0.0013229, 0.0013351
5: 0.0078780, 0.0096012, 0.0078997, 0.0095337, -0.0013358, 0.0013733
6: 0.0093097, 0.0099600, 0.0093351, 0.0099518, -0.0006421, 0.0006248
7: -0.0192427, -0.0155018, -0.0190962, -0.0155488, -0.0028940, 0.0028018
8: 0.9686583, 0.9793764, 0.9690781, 0.9792416, -0.0086154, 0.0083990
9: 0.0037144, 0.0068645, 0.0037540, 0.0067411, -0.0023887, 0.0024644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057761, upper bound: 0.0057743
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059911, upper bound: 0.0058335
time: 0.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002832, 0.0001344, -0.0002832, 0.0001344, -0.0003178, 0.0003178
1: -0.0000054, 0.0015277, -0.0000054, 0.0015277, -0.0012165, 0.0012165
2: 0.0140522, 0.0163480, 0.0140522, 0.0163480, -0.0017932, 0.0017932
3: -0.0000603, 0.0016661, -0.0000603, 0.0016661, -0.0013358, 0.0013358
4: -0.0044352, -0.0028428, -0.0044352, -0.0028428, -0.0013367, 0.0013367
5: 0.0078780, 0.0096012, 0.0078780, 0.0096012, -0.0013322, 0.0013322
6: 0.0093097, 0.0099600, 0.0093097, 0.0099600, -0.0006503, 0.0006503
7: -0.0192427, -0.0155018, -0.0192427, -0.0155018, -0.0027698, 0.0027698
8: 0.9686583, 0.9793764, 0.9686583, 0.9793764, -0.0084012, 0.0084012
9: 0.0037144, 0.0068645, 0.0037144, 0.0068645, -0.0023694, 0.0023694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059176, upper bound: 0.0055944
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059911, upper bound: 0.0058335
time: 0.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 8, lower bound: -0.0057777, upper bound: 0.0055988
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 8, lower bound: -0.0058348, upper bound: 0.0058335
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 8, lower bound: -0.0057777, upper bound: 0.0057761
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 8, lower bound: -0.0058348, upper bound: 0.0059911
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 8, lower bound: -0.0057761, upper bound: 0.0057743
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 8, lower bound: -0.0059911, upper bound: 0.0058335
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 8, lower bound: -0.0059176, upper bound: 0.0055944
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 8, lower bound: -0.0059911, upper bound: 0.0058335

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002762, 0.0000599, -0.0002699, 0.0001119, -0.0002848, 0.0002368
1: 0.0000274, 0.0014134, 0.0000566, 0.0014931, -0.0010523, 0.0010233
2: 0.0142233, 0.0162990, 0.0141038, 0.0162553, -0.0015119, 0.0015672
3: 0.0000684, 0.0016292, -0.0000214, 0.0015964, -0.0011281, 0.0011746
4: -0.0043166, -0.0028768, -0.0043994, -0.0029071, -0.0011184, 0.0011167
5: 0.0080064, 0.0095644, 0.0079168, 0.0095317, -0.0011252, 0.0011721
6: 0.0093236, 0.0099115, 0.0093359, 0.0099453, -0.0005515, 0.0005756
7: -0.0191628, -0.0157805, -0.0190916, -0.0155860, -0.0025010, 0.0023547
8: 0.9688872, 0.9785777, 0.9690911, 0.9791352, -0.0073254, 0.0070790
9: 0.0039492, 0.0067972, 0.0037853, 0.0067373, -0.0020095, 0.0021199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0053506
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055656, upper bound: 0.0053504
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001046, -0.0002703, 0.0001218, -0.0002981, 0.0002733
1: 0.0000573, 0.0014820, 0.0000547, 0.0015084, -0.0011289, 0.0010761
2: 0.0141206, 0.0162542, 0.0140810, 0.0162581, -0.0015865, 0.0016719
3: -0.0000088, 0.0015956, -0.0000385, 0.0015985, -0.0011821, 0.0012490
4: -0.0043878, -0.0029079, -0.0044152, -0.0029052, -0.0011869, 0.0012229
5: 0.0079294, 0.0095308, 0.0078997, 0.0095337, -0.0011789, 0.0012460
6: 0.0093362, 0.0099406, 0.0093351, 0.0099518, -0.0006155, 0.0006054
7: -0.0190899, -0.0156132, -0.0190962, -0.0155488, -0.0026239, 0.0024526
8: 0.9690961, 0.9790571, 0.9690781, 0.9792416, -0.0078240, 0.0074315
9: 0.0038083, 0.0067358, 0.0037540, 0.0067411, -0.0020977, 0.0022343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056310, upper bound: 0.0056607
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056607, upper bound: 0.0056607
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002762, 0.0000599, -0.0002828, 0.0001247, -0.0003081, 0.0002597
1: 0.0000274, 0.0014134, -0.0000035, 0.0015127, -0.0011240, 0.0011231
2: 0.0142233, 0.0162990, 0.0140746, 0.0163453, -0.0016654, 0.0016746
3: 0.0000684, 0.0016292, -0.0000434, 0.0016641, -0.0012450, 0.0012554
4: -0.0043166, -0.0028768, -0.0044197, -0.0028447, -0.0012069, 0.0011912
5: 0.0080064, 0.0095644, 0.0078948, 0.0095992, -0.0012420, 0.0012527
6: 0.0093236, 0.0099115, 0.0093104, 0.0099536, -0.0005819, 0.0006011
7: -0.0191628, -0.0157805, -0.0192382, -0.0155383, -0.0026761, 0.0026160
8: 0.9688872, 0.9785777, 0.9686710, 0.9792719, -0.0078269, 0.0077920
9: 0.0039492, 0.0067972, 0.0037451, 0.0068607, -0.0022281, 0.0022673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0055278
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055656, upper bound: 0.0055259
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001046, -0.0002832, 0.0001344, -0.0003216, 0.0002985
1: 0.0000573, 0.0014820, -0.0000054, 0.0015277, -0.0011980, 0.0011805
2: 0.0141206, 0.0162542, 0.0140522, 0.0163480, -0.0017481, 0.0017754
3: -0.0000088, 0.0015956, -0.0000603, 0.0016661, -0.0013060, 0.0013269
4: -0.0043878, -0.0029079, -0.0044352, -0.0028428, -0.0012798, 0.0012947
5: 0.0079294, 0.0095308, 0.0078780, 0.0096012, -0.0013028, 0.0013237
6: 0.0093362, 0.0099406, 0.0093097, 0.0099600, -0.0006237, 0.0006309
7: -0.0190899, -0.0156132, -0.0192427, -0.0155018, -0.0027926, 0.0027378
8: 0.9690961, 0.9790571, 0.9686583, 0.9793764, -0.0083074, 0.0081814
9: 0.0038083, 0.0067358, 0.0037144, 0.0068645, -0.0023343, 0.0023763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056310, upper bound: 0.0058219
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056607, upper bound: 0.0058219
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0002828, 0.0001247, -0.0002762, 0.0000599, -0.0002597, 0.0003081
1: -0.0000035, 0.0015127, 0.0000274, 0.0014134, -0.0011231, 0.0011240
2: 0.0140746, 0.0163453, 0.0142233, 0.0162990, -0.0016746, 0.0016654
3: -0.0000434, 0.0016641, 0.0000684, 0.0016292, -0.0012554, 0.0012450
4: -0.0044197, -0.0028447, -0.0043166, -0.0028768, -0.0011912, 0.0012069
5: 0.0078948, 0.0095992, 0.0080064, 0.0095644, -0.0012527, 0.0012420
6: 0.0093104, 0.0099536, 0.0093236, 0.0099115, -0.0006011, 0.0005819
7: -0.0192382, -0.0155383, -0.0191628, -0.0157805, -0.0026160, 0.0026761
8: 0.9686710, 0.9792719, 0.9688872, 0.9785777, -0.0077920, 0.0078269
9: 0.0037451, 0.0068607, 0.0039492, 0.0067972, -0.0022673, 0.0022281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055278, upper bound: 0.0055508
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0055656
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0002832, 0.0001344, -0.0002698, 0.0001046, -0.0002985, 0.0003216
1: -0.0000054, 0.0015277, 0.0000573, 0.0014820, -0.0011805, 0.0011980
2: 0.0140522, 0.0163480, 0.0141206, 0.0162542, -0.0017754, 0.0017481
3: -0.0000603, 0.0016661, -0.0000088, 0.0015956, -0.0013269, 0.0013060
4: -0.0044352, -0.0028428, -0.0043878, -0.0029079, -0.0012947, 0.0012798
5: 0.0078780, 0.0096012, 0.0079294, 0.0095308, -0.0013237, 0.0013028
6: 0.0093097, 0.0099600, 0.0093362, 0.0099406, -0.0006309, 0.0006237
7: -0.0192427, -0.0155018, -0.0190899, -0.0156132, -0.0027378, 0.0027926
8: 0.9686583, 0.9793764, 0.9690961, 0.9790571, -0.0081814, 0.0083074
9: 0.0037144, 0.0068645, 0.0038083, 0.0067358, -0.0023763, 0.0023343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058219, upper bound: 0.0056310
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058219, upper bound: 0.0056607
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002898, 0.0000758, -0.0002828, 0.0001247, -0.0003039, 0.0002582
1: -0.0000364, 0.0014377, -0.0000035, 0.0015127, -0.0011203, 0.0010942
2: 0.0141868, 0.0163944, 0.0140746, 0.0163453, -0.0016125, 0.0016641
3: 0.0000410, 0.0017010, -0.0000434, 0.0016641, -0.0012011, 0.0012454
4: -0.0043418, -0.0028106, -0.0044197, -0.0028447, -0.0012030, 0.0011985
5: 0.0079791, 0.0096361, 0.0078948, 0.0095992, -0.0011979, 0.0012426
6: 0.0092965, 0.0099218, 0.0093104, 0.0099536, -0.0006237, 0.0006114
7: -0.0193183, -0.0157212, -0.0192382, -0.0155383, -0.0026351, 0.0024892
8: 0.9684415, 0.9787478, 0.9686710, 0.9792719, -0.0077826, 0.0075545
9: 0.0038992, 0.0069282, 0.0037451, 0.0068607, -0.0021300, 0.0022391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056512, upper bound: 0.0053482
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056906, upper bound: 0.0053479
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002826, 0.0001164, -0.0002832, 0.0001344, -0.0003172, 0.0002922
1: -0.0000029, 0.0015001, -0.0000054, 0.0015277, -0.0011990, 0.0011494
2: 0.0140935, 0.0163443, 0.0140522, 0.0163480, -0.0016886, 0.0017715
3: -0.0000292, 0.0016633, -0.0000603, 0.0016661, -0.0012556, 0.0013218
4: -0.0044066, -0.0028454, -0.0044352, -0.0028428, -0.0012742, 0.0013067
5: 0.0079090, 0.0095984, 0.0078780, 0.0096012, -0.0012520, 0.0013185
6: 0.0093107, 0.0099483, 0.0093097, 0.0099600, -0.0006492, 0.0006386
7: -0.0192366, -0.0155691, -0.0192427, -0.0155018, -0.0027598, 0.0025877
8: 0.9686757, 0.9791836, 0.9686583, 0.9793764, -0.0082951, 0.0079163
9: 0.0037711, 0.0068594, 0.0037144, 0.0068645, -0.0022181, 0.0023551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057716, upper bound: 0.0056601
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058219, upper bound: 0.0056601
time: 0.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.55 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0053506
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0055656, upper bound: 0.0053504
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0056310, upper bound: 0.0056607
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0056607, upper bound: 0.0056607
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0055278
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0055656, upper bound: 0.0055259
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0056310, upper bound: 0.0058219
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0056607, upper bound: 0.0058219
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0055278, upper bound: 0.0055508
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0055656
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0058219, upper bound: 0.0056310
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0058219, upper bound: 0.0056607
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0056512, upper bound: 0.0053482
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0056906, upper bound: 0.0053479
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0057716, upper bound: 0.0056601
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 8, lower bound: -0.0058219, upper bound: 0.0056601

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002762, 0.0000598, -0.0002695, 0.0000882, -0.0002621, 0.0002351
1: 0.0000274, 0.0014133, 0.0000586, 0.0014568, -0.0010159, 0.0009952
2: 0.0142234, 0.0162990, 0.0141583, 0.0162522, -0.0014764, 0.0015128
3: 0.0000685, 0.0016292, 0.0000195, 0.0015941, -0.0011042, 0.0011337
4: -0.0043165, -0.0028768, -0.0043616, -0.0029092, -0.0010719, 0.0010789
5: 0.0080065, 0.0095644, 0.0079577, 0.0095294, -0.0011016, 0.0011313
6: 0.0093236, 0.0099115, 0.0093368, 0.0099299, -0.0005359, 0.0005747
7: -0.0191628, -0.0157808, -0.0190866, -0.0156747, -0.0024125, 0.0023301
8: 0.9688872, 0.9785770, 0.9691055, 0.9788811, -0.0070716, 0.0069071
9: 0.0039494, 0.0067972, 0.0038600, 0.0067331, -0.0019814, 0.0020453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0053265
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0053504
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002761, 0.0000562, -0.0002736, 0.0000865, -0.0002619, 0.0002376
1: 0.0000278, 0.0014078, 0.0000393, 0.0014542, -0.0010184, 0.0010174
2: 0.0142316, 0.0162984, 0.0141622, 0.0162811, -0.0015097, 0.0015172
3: 0.0000747, 0.0016288, 0.0000225, 0.0016158, -0.0011289, 0.0011373
4: -0.0043107, -0.0028772, -0.0043589, -0.0028892, -0.0010909, 0.0010792
5: 0.0080127, 0.0095640, 0.0079606, 0.0095510, -0.0011262, 0.0011349
6: 0.0093237, 0.0099091, 0.0093286, 0.0099288, -0.0005281, 0.0005805
7: -0.0191618, -0.0157942, -0.0191336, -0.0156810, -0.0024237, 0.0023822
8: 0.9688900, 0.9785386, 0.9689708, 0.9788629, -0.0070914, 0.0070625
9: 0.0039607, 0.0067964, 0.0038654, 0.0067726, -0.0020251, 0.0020536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053504, upper bound: 0.0053504
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053504, upper bound: 0.0053504
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001046, -0.0002699, 0.0000984, -0.0002762, 0.0002715
1: 0.0000573, 0.0014819, 0.0000567, 0.0014725, -0.0010913, 0.0010459
2: 0.0141207, 0.0162542, 0.0141348, 0.0162550, -0.0015480, 0.0016156
3: -0.0000087, 0.0015956, 0.0000019, 0.0015962, -0.0011563, 0.0012067
4: -0.0043877, -0.0029079, -0.0043779, -0.0029073, -0.0011369, 0.0011838
5: 0.0079295, 0.0095308, 0.0079400, 0.0095315, -0.0011535, 0.0012037
6: 0.0093362, 0.0099405, 0.0093360, 0.0099366, -0.0006003, 0.0006045
7: -0.0190899, -0.0156135, -0.0190912, -0.0156364, -0.0025323, 0.0024252
8: 0.9690962, 0.9790565, 0.9690922, 0.9789908, -0.0075612, 0.0072452
9: 0.0038085, 0.0067358, 0.0038278, 0.0067369, -0.0020660, 0.0021571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0055656
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0056607
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002697, 0.0001010, -0.0002740, 0.0000966, -0.0002763, 0.0002735
1: 0.0000577, 0.0014764, 0.0000375, 0.0014696, -0.0010960, 0.0010660
2: 0.0141290, 0.0162536, 0.0141391, 0.0162838, -0.0015780, 0.0016235
3: -0.0000025, 0.0015952, 0.0000051, 0.0016178, -0.0011784, 0.0012130
4: -0.0043819, -0.0029083, -0.0043749, -0.0028873, -0.0011531, 0.0011857
5: 0.0079357, 0.0095304, 0.0079433, 0.0095531, -0.0011755, 0.0012100
6: 0.0093364, 0.0099382, 0.0093279, 0.0099353, -0.0005989, 0.0006103
7: -0.0190889, -0.0156269, -0.0191381, -0.0156434, -0.0025500, 0.0024738
8: 0.9690988, 0.9790179, 0.9689581, 0.9789706, -0.0075973, 0.0073861
9: 0.0038198, 0.0067350, 0.0038337, 0.0067764, -0.0021069, 0.0021708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053504, upper bound: 0.0055656
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053504, upper bound: 0.0056607
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002762, 0.0000598, -0.0002822, 0.0001003, -0.0002847, 0.0002578
1: 0.0000274, 0.0014133, -0.0000011, 0.0014753, -0.0010878, 0.0010991
2: 0.0142234, 0.0162990, 0.0141305, 0.0163416, -0.0016340, 0.0016205
3: 0.0000685, 0.0016292, -0.0000014, 0.0016613, -0.0012232, 0.0012147
4: -0.0043165, -0.0028768, -0.0043809, -0.0028473, -0.0011712, 0.0011536
5: 0.0080065, 0.0095644, 0.0079368, 0.0095964, -0.0012204, 0.0012121
6: 0.0093236, 0.0099115, 0.0093115, 0.0099378, -0.0005664, 0.0006000
7: -0.0191628, -0.0157808, -0.0192322, -0.0156294, -0.0025880, 0.0025895
8: 0.9688872, 0.9785770, 0.9686883, 0.9790108, -0.0075743, 0.0076409
9: 0.0039494, 0.0067972, 0.0038219, 0.0068557, -0.0021993, 0.0021931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0055009
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0055259
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002761, 0.0000562, -0.0002873, 0.0001003, -0.0002856, 0.0002591
1: 0.0000278, 0.0014078, -0.0000247, 0.0014754, -0.0010901, 0.0011134
2: 0.0142316, 0.0162984, 0.0141304, 0.0163770, -0.0016556, 0.0016246
3: 0.0000747, 0.0016288, -0.0000014, 0.0016879, -0.0012396, 0.0012180
4: -0.0043107, -0.0028772, -0.0043809, -0.0028227, -0.0011883, 0.0011537
5: 0.0080127, 0.0095640, 0.0079368, 0.0096230, -0.0012369, 0.0012155
6: 0.0093237, 0.0099091, 0.0093015, 0.0099378, -0.0005585, 0.0006077
7: -0.0191618, -0.0157942, -0.0192899, -0.0156293, -0.0025986, 0.0026269
8: 0.9688900, 0.9785386, 0.9685231, 0.9790111, -0.0075926, 0.0077409
9: 0.0039607, 0.0067964, 0.0038218, 0.0069042, -0.0022308, 0.0022009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053479, upper bound: 0.0055259
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053479, upper bound: 0.0055259
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001046, -0.0002826, 0.0001100, -0.0002982, 0.0002967
1: 0.0000573, 0.0014819, -0.0000029, 0.0014902, -0.0011578, 0.0011554
2: 0.0141207, 0.0162542, 0.0141082, 0.0163444, -0.0017156, 0.0017152
3: -0.0000087, 0.0015956, -0.0000181, 0.0016634, -0.0012835, 0.0012816
4: -0.0043877, -0.0029079, -0.0043963, -0.0028453, -0.0012383, 0.0012529
5: 0.0079295, 0.0095308, 0.0079201, 0.0095985, -0.0012806, 0.0012785
6: 0.0093362, 0.0099405, 0.0093107, 0.0099441, -0.0006078, 0.0006298
7: -0.0190899, -0.0156135, -0.0192367, -0.0155931, -0.0026946, 0.0027122
8: 0.9690962, 0.9790565, 0.9686753, 0.9791148, -0.0080263, 0.0080241
9: 0.0038085, 0.0067358, 0.0037913, 0.0068595, -0.0023055, 0.0022938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0056906
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0058219
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002697, 0.0001010, -0.0002877, 0.0001100, -0.0002995, 0.0002992
1: 0.0000577, 0.0014764, -0.0000265, 0.0014903, -0.0011616, 0.0011754
2: 0.0141290, 0.0162536, 0.0141081, 0.0163797, -0.0017469, 0.0017217
3: -0.0000025, 0.0015952, -0.0000182, 0.0016900, -0.0013079, 0.0012868
4: -0.0043819, -0.0029083, -0.0043964, -0.0028208, -0.0012571, 0.0012538
5: 0.0079357, 0.0095304, 0.0079200, 0.0096250, -0.0013050, 0.0012837
6: 0.0093364, 0.0099382, 0.0093007, 0.0099441, -0.0006077, 0.0006375
7: -0.0190889, -0.0156269, -0.0192944, -0.0155930, -0.0027100, 0.0027660
8: 0.9690988, 0.9790179, 0.9685102, 0.9791152, -0.0080559, 0.0081692
9: 0.0038198, 0.0067350, 0.0037912, 0.0069080, -0.0023515, 0.0023056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053479, upper bound: 0.0056906
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053479, upper bound: 0.0058219
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002822, 0.0001003, -0.0002762, 0.0000598, -0.0002578, 0.0002847
1: -0.0000011, 0.0014753, 0.0000274, 0.0014133, -0.0010991, 0.0010878
2: 0.0141305, 0.0163416, 0.0142234, 0.0162990, -0.0016205, 0.0016340
3: -0.0000014, 0.0016613, 0.0000685, 0.0016292, -0.0012147, 0.0012232
4: -0.0043809, -0.0028473, -0.0043165, -0.0028768, -0.0011536, 0.0011712
5: 0.0079368, 0.0095964, 0.0080065, 0.0095644, -0.0012121, 0.0012204
6: 0.0093115, 0.0099378, 0.0093236, 0.0099115, -0.0006000, 0.0005664
7: -0.0192322, -0.0156294, -0.0191628, -0.0157808, -0.0025895, 0.0025880
8: 0.9686883, 0.9790108, 0.9688872, 0.9785770, -0.0076409, 0.0075743
9: 0.0038219, 0.0068557, 0.0039494, 0.0067972, -0.0021931, 0.0021993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055508
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055508
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002873, 0.0001003, -0.0002761, 0.0000562, -0.0002591, 0.0002856
1: -0.0000247, 0.0014754, 0.0000278, 0.0014078, -0.0011134, 0.0010901
2: 0.0141304, 0.0163770, 0.0142316, 0.0162984, -0.0016246, 0.0016556
3: -0.0000014, 0.0016879, 0.0000747, 0.0016288, -0.0012180, 0.0012396
4: -0.0043809, -0.0028227, -0.0043107, -0.0028772, -0.0011537, 0.0011883
5: 0.0079368, 0.0096230, 0.0080127, 0.0095640, -0.0012155, 0.0012369
6: 0.0093015, 0.0099378, 0.0093237, 0.0099091, -0.0006077, 0.0005585
7: -0.0192899, -0.0156293, -0.0191618, -0.0157942, -0.0026269, 0.0025986
8: 0.9685231, 0.9790111, 0.9688900, 0.9785386, -0.0077409, 0.0075926
9: 0.0038218, 0.0069042, 0.0039607, 0.0067964, -0.0022009, 0.0022308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0053479
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0055656
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002826, 0.0001100, -0.0002698, 0.0001046, -0.0002967, 0.0002982
1: -0.0000029, 0.0014902, 0.0000573, 0.0014819, -0.0011554, 0.0011578
2: 0.0141082, 0.0163444, 0.0141207, 0.0162542, -0.0017152, 0.0017156
3: -0.0000181, 0.0016634, -0.0000087, 0.0015956, -0.0012816, 0.0012835
4: -0.0043963, -0.0028453, -0.0043877, -0.0029079, -0.0012529, 0.0012383
5: 0.0079201, 0.0095985, 0.0079295, 0.0095308, -0.0012785, 0.0012806
6: 0.0093107, 0.0099441, 0.0093362, 0.0099405, -0.0006298, 0.0006078
7: -0.0192367, -0.0155931, -0.0190899, -0.0156135, -0.0027122, 0.0026946
8: 0.9686753, 0.9791148, 0.9690962, 0.9790565, -0.0080241, 0.0080263
9: 0.0037913, 0.0068595, 0.0038085, 0.0067358, -0.0022938, 0.0023055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055278, upper bound: 0.0053245
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055278, upper bound: 0.0056310
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002877, 0.0001100, -0.0002697, 0.0001010, -0.0002992, 0.0002995
1: -0.0000265, 0.0014903, 0.0000577, 0.0014764, -0.0011754, 0.0011616
2: 0.0141081, 0.0163797, 0.0141290, 0.0162536, -0.0017217, 0.0017469
3: -0.0000182, 0.0016900, -0.0000025, 0.0015952, -0.0012868, 0.0013079
4: -0.0043964, -0.0028208, -0.0043819, -0.0029083, -0.0012538, 0.0012571
5: 0.0079200, 0.0096250, 0.0079357, 0.0095304, -0.0012837, 0.0013050
6: 0.0093007, 0.0099441, 0.0093364, 0.0099382, -0.0006375, 0.0006077
7: -0.0192944, -0.0155930, -0.0190889, -0.0156269, -0.0027660, 0.0027100
8: 0.9685102, 0.9791152, 0.9690988, 0.9790179, -0.0081692, 0.0080559
9: 0.0037912, 0.0069080, 0.0038198, 0.0067350, -0.0023056, 0.0023515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0053479
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0056607
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002898, 0.0000757, -0.0002822, 0.0001003, -0.0002805, 0.0002566
1: -0.0000363, 0.0014376, -0.0000011, 0.0014753, -0.0010838, 0.0010631
2: 0.0141870, 0.0163944, 0.0141305, 0.0163416, -0.0015743, 0.0016094
3: 0.0000411, 0.0017010, -0.0000014, 0.0016613, -0.0011759, 0.0012043
4: -0.0043417, -0.0028106, -0.0043809, -0.0028473, -0.0011540, 0.0011605
5: 0.0079792, 0.0096361, 0.0079368, 0.0095964, -0.0011730, 0.0012015
6: 0.0092965, 0.0099218, 0.0093115, 0.0099378, -0.0006081, 0.0006103
7: -0.0193183, -0.0157214, -0.0192322, -0.0156294, -0.0025461, 0.0024642
8: 0.9684415, 0.9787471, 0.9686883, 0.9790108, -0.0075273, 0.0073679
9: 0.0038994, 0.0069282, 0.0038219, 0.0068557, -0.0021001, 0.0021641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056512, upper bound: 0.0053245
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056512, upper bound: 0.0053479
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002897, 0.0000723, -0.0002873, 0.0001003, -0.0002820, 0.0002590
1: -0.0000359, 0.0014323, -0.0000247, 0.0014754, -0.0010883, 0.0010846
2: 0.0141949, 0.0163937, 0.0141304, 0.0163770, -0.0016068, 0.0016173
3: 0.0000471, 0.0017005, -0.0000014, 0.0016879, -0.0012002, 0.0012106
4: -0.0043362, -0.0028111, -0.0043809, -0.0028227, -0.0011725, 0.0011632
5: 0.0079851, 0.0096355, 0.0079368, 0.0096230, -0.0011972, 0.0012079
6: 0.0092967, 0.0099195, 0.0093015, 0.0099378, -0.0006014, 0.0006181
7: -0.0193172, -0.0157343, -0.0192899, -0.0156293, -0.0025638, 0.0025140
8: 0.9684449, 0.9787102, 0.9685231, 0.9790111, -0.0075630, 0.0075194
9: 0.0039102, 0.0069272, 0.0038218, 0.0069042, -0.0021426, 0.0021778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0053479
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0053479
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002826, 0.0001164, -0.0002826, 0.0001100, -0.0002942, 0.0002903
1: -0.0000029, 0.0015000, -0.0000029, 0.0014902, -0.0011609, 0.0011153
2: 0.0140936, 0.0163443, 0.0141082, 0.0163444, -0.0016463, 0.0017145
3: -0.0000291, 0.0016633, -0.0000181, 0.0016634, -0.0012279, 0.0012790
4: -0.0044065, -0.0028454, -0.0043963, -0.0028453, -0.0012207, 0.0012671
5: 0.0079091, 0.0095984, 0.0079201, 0.0095985, -0.0012247, 0.0012757
6: 0.0093107, 0.0099482, 0.0093107, 0.0099441, -0.0006334, 0.0006375
7: -0.0192366, -0.0155693, -0.0192367, -0.0155931, -0.0026670, 0.0025601
8: 0.9686757, 0.9791829, 0.9686753, 0.9791148, -0.0080290, 0.0077100
9: 0.0037713, 0.0068594, 0.0037913, 0.0068595, -0.0021864, 0.0022770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055623
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056601
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002825, 0.0001129, -0.0002877, 0.0001100, -0.0002961, 0.0002924
1: -0.0000024, 0.0014946, -0.0000265, 0.0014903, -0.0011680, 0.0011362
2: 0.0141016, 0.0163436, 0.0141081, 0.0163797, -0.0016791, 0.0017262
3: -0.0000231, 0.0016628, -0.0000182, 0.0016900, -0.0012529, 0.0012882
4: -0.0044009, -0.0028459, -0.0043964, -0.0028208, -0.0012377, 0.0012719
5: 0.0079151, 0.0095979, 0.0079200, 0.0096250, -0.0012497, 0.0012850
6: 0.0093109, 0.0099460, 0.0093007, 0.0099441, -0.0006332, 0.0006453
7: -0.0192355, -0.0155824, -0.0192944, -0.0155930, -0.0026912, 0.0026088
8: 0.9686789, 0.9791455, 0.9685102, 0.9791152, -0.0080827, 0.0078615
9: 0.0037823, 0.0068584, 0.0037912, 0.0069080, -0.0022282, 0.0022961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0055623
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0056601
time: 0.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0053265
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0053504
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053504, upper bound: 0.0053504
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053504, upper bound: 0.0053504
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0055656
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0056607
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053504, upper bound: 0.0055656
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053504, upper bound: 0.0056607
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0055009
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055508, upper bound: 0.0055259
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053479, upper bound: 0.0055259
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053479, upper bound: 0.0055259
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0056906
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0058219
IS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053479, upper bound: 0.0056906
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0053479, upper bound: 0.0058219
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055508
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055508
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0053479
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0055656
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055278, upper bound: 0.0053245
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055278, upper bound: 0.0056310
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0053479
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0056607
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0056512, upper bound: 0.0053245
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0056512, upper bound: 0.0053479
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0053479
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0053479
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055623
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056601
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0055623
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 8, lower bound: -0.0055259, upper bound: 0.0056601

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002695, 0.0000882, -0.0002612, 0.0002133
1: 0.0000295, 0.0013767, 0.0000586, 0.0014568, -0.0009971, 0.0009576
2: 0.0142782, 0.0162958, 0.0141583, 0.0162522, -0.0014202, 0.0014886
3: 0.0001097, 0.0016268, 0.0000195, 0.0015941, -0.0010619, 0.0011172
4: -0.0042785, -0.0028790, -0.0043616, -0.0029092, -0.0010329, 0.0010465
5: 0.0080477, 0.0095620, 0.0079577, 0.0095294, -0.0010594, 0.0011149
6: 0.0093245, 0.0098959, 0.0093368, 0.0099299, -0.0004729, 0.0005591
7: -0.0191576, -0.0158701, -0.0190866, -0.0156747, -0.0023969, 0.0022385
8: 0.9689021, 0.9783213, 0.9691055, 0.9788811, -0.0069543, 0.0066447
9: 0.0040245, 0.0067928, 0.0038600, 0.0067331, -0.0019043, 0.0020258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053413
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053413
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002695, 0.0000882, -0.0002669, 0.0002135
1: 0.0000120, 0.0013742, 0.0000586, 0.0014568, -0.0010223, 0.0009609
2: 0.0142819, 0.0163219, 0.0141583, 0.0162522, -0.0014252, 0.0015267
3: 0.0001125, 0.0016465, 0.0000195, 0.0015941, -0.0010656, 0.0011462
4: -0.0042759, -0.0028609, -0.0043616, -0.0029092, -0.0010363, 0.0010723
5: 0.0080505, 0.0095817, 0.0079577, 0.0095294, -0.0010631, 0.0011439
6: 0.0093171, 0.0098949, 0.0093368, 0.0099299, -0.0004785, 0.0005581
7: -0.0192002, -0.0158761, -0.0190866, -0.0156747, -0.0024624, 0.0022466
8: 0.9687800, 0.9783039, 0.9691055, 0.9788811, -0.0071321, 0.0066677
9: 0.0040296, 0.0068287, 0.0038600, 0.0067331, -0.0019111, 0.0020804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053506
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053506
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002761, 0.0000562, -0.0002797, 0.0000343, -0.0002104, 0.0002385
1: 0.0000278, 0.0014078, 0.0000107, 0.0013742, -0.0009261, 0.0009906
2: 0.0142316, 0.0162984, 0.0142819, 0.0163240, -0.0014790, 0.0013790
3: 0.0000747, 0.0016288, 0.0001125, 0.0016480, -0.0011101, 0.0010334
4: -0.0043107, -0.0028772, -0.0042759, -0.0028595, -0.0010399, 0.0009833
5: 0.0080127, 0.0095640, 0.0080505, 0.0095832, -0.0011079, 0.0010311
6: 0.0093237, 0.0099091, 0.0093165, 0.0098949, -0.0004889, 0.0004680
7: -0.0191618, -0.0157942, -0.0192035, -0.0158761, -0.0021984, 0.0023824
8: 0.9688900, 0.9785386, 0.9687706, 0.9783039, -0.0064460, 0.0069093
9: 0.0039607, 0.0067964, 0.0040296, 0.0068315, -0.0020135, 0.0018639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053265
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053504
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002761, 0.0000562, -0.0002734, 0.0000794, -0.0002585, 0.0002374
1: 0.0000278, 0.0014078, 0.0000401, 0.0014433, -0.0010258, 0.0010118
2: 0.0142316, 0.0162984, 0.0141785, 0.0162800, -0.0015024, 0.0015283
3: 0.0000747, 0.0016288, 0.0000347, 0.0016150, -0.0011241, 0.0011457
4: -0.0043107, -0.0028772, -0.0043476, -0.0028900, -0.0010818, 0.0010869
5: 0.0080127, 0.0095640, 0.0079728, 0.0095502, -0.0011215, 0.0011432
6: 0.0093237, 0.0099091, 0.0093289, 0.0099242, -0.0005312, 0.0005692
7: -0.0191618, -0.0157942, -0.0191318, -0.0157076, -0.0024417, 0.0023784
8: 0.9688900, 0.9785386, 0.9689760, 0.9787869, -0.0071431, 0.0070274
9: 0.0039607, 0.0067964, 0.0038877, 0.0067711, -0.0020204, 0.0020688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053265
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053504
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001046, -0.0002757, 0.0000360, -0.0002148, 0.0002801
1: 0.0000573, 0.0014819, 0.0000295, 0.0013767, -0.0009794, 0.0010410
2: 0.0141207, 0.0162542, 0.0142782, 0.0162958, -0.0015544, 0.0014481
3: -0.0000087, 0.0015956, 0.0001097, 0.0016268, -0.0011667, 0.0010807
4: -0.0043877, -0.0029079, -0.0042785, -0.0028790, -0.0010922, 0.0010676
5: 0.0079295, 0.0095308, 0.0080477, 0.0095620, -0.0011644, 0.0010780
6: 0.0093362, 0.0099405, 0.0093245, 0.0098959, -0.0005597, 0.0004916
7: -0.0190899, -0.0156135, -0.0191576, -0.0158701, -0.0022593, 0.0025042
8: 0.9690962, 0.9790565, 0.9689021, 0.9783213, -0.0067792, 0.0072617
9: 0.0038085, 0.0067358, 0.0040245, 0.0067928, -0.0021162, 0.0019273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0055550
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0055656
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001046, -0.0002693, 0.0000816, -0.0002508, 0.0002709
1: 0.0000573, 0.0014819, 0.0000593, 0.0014467, -0.0010214, 0.0010304
2: 0.0141207, 0.0162542, 0.0141734, 0.0162511, -0.0015294, 0.0015091
3: -0.0000087, 0.0015956, 0.0000309, 0.0015933, -0.0011439, 0.0011262
4: -0.0043877, -0.0029079, -0.0043511, -0.0029100, -0.0011087, 0.0011168
5: 0.0079295, 0.0095308, 0.0079690, 0.0095285, -0.0011413, 0.0011234
6: 0.0093362, 0.0099405, 0.0093371, 0.0099256, -0.0005894, 0.0005999
7: -0.0190899, -0.0156135, -0.0190849, -0.0156993, -0.0023502, 0.0024151
8: 0.9690962, 0.9790565, 0.9691105, 0.9788105, -0.0070664, 0.0071546
9: 0.0038085, 0.0067358, 0.0038807, 0.0067316, -0.0020531, 0.0020058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0056347
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0056607
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002697, 0.0001010, -0.0002797, 0.0000343, -0.0002153, 0.0002857
1: 0.0000577, 0.0014764, 0.0000107, 0.0013742, -0.0009845, 0.0010905
2: 0.0141290, 0.0162536, 0.0142819, 0.0163240, -0.0016286, 0.0014565
3: -0.0000025, 0.0015952, 0.0001125, 0.0016480, -0.0012226, 0.0010874
4: -0.0043819, -0.0029083, -0.0042759, -0.0028595, -0.0011436, 0.0010699
5: 0.0079357, 0.0095304, 0.0080505, 0.0095832, -0.0012202, 0.0010847
6: 0.0093364, 0.0099382, 0.0093165, 0.0098949, -0.0005585, 0.0005104
7: -0.0190889, -0.0156269, -0.0192035, -0.0158761, -0.0022778, 0.0026262
8: 0.9690988, 0.9790179, 0.9687706, 0.9783039, -0.0068176, 0.0076079
9: 0.0038198, 0.0067350, 0.0040296, 0.0068315, -0.0022188, 0.0019416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0055508
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0055656
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002697, 0.0001010, -0.0002734, 0.0000794, -0.0002506, 0.0002729
1: 0.0000577, 0.0014764, 0.0000401, 0.0014433, -0.0010254, 0.0010499
2: 0.0141290, 0.0162536, 0.0141785, 0.0162800, -0.0015580, 0.0015162
3: -0.0000025, 0.0015952, 0.0000347, 0.0016150, -0.0011653, 0.0011321
4: -0.0043819, -0.0029083, -0.0043476, -0.0028900, -0.0011266, 0.0011178
5: 0.0079357, 0.0095304, 0.0079728, 0.0095502, -0.0011626, 0.0011293
6: 0.0093364, 0.0099382, 0.0093289, 0.0099242, -0.0005878, 0.0005964
7: -0.0190889, -0.0156269, -0.0191318, -0.0157076, -0.0023672, 0.0024632
8: 0.9690988, 0.9790179, 0.9689760, 0.9787869, -0.0070979, 0.0072879
9: 0.0038198, 0.0067350, 0.0038877, 0.0067711, -0.0020937, 0.0020189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0056310
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0056607
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002822, 0.0001003, -0.0002838, 0.0002361
1: 0.0000295, 0.0013767, -0.0000011, 0.0014753, -0.0010690, 0.0010615
2: 0.0142782, 0.0162958, 0.0141305, 0.0163416, -0.0015778, 0.0015963
3: 0.0001097, 0.0016268, -0.0000014, 0.0016613, -0.0011809, 0.0011982
4: -0.0042785, -0.0028790, -0.0043809, -0.0028473, -0.0011322, 0.0011212
5: 0.0080477, 0.0095620, 0.0079368, 0.0095964, -0.0011782, 0.0011958
6: 0.0093245, 0.0098959, 0.0093115, 0.0099378, -0.0005034, 0.0005844
7: -0.0191576, -0.0158701, -0.0192322, -0.0156294, -0.0025724, 0.0024979
8: 0.9689021, 0.9783213, 0.9686883, 0.9790108, -0.0074570, 0.0073785
9: 0.0040245, 0.0067928, 0.0038219, 0.0068557, -0.0021222, 0.0021736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055174
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055174
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002822, 0.0001003, -0.0002895, 0.0002362
1: 0.0000120, 0.0013742, -0.0000011, 0.0014753, -0.0010943, 0.0010648
2: 0.0142819, 0.0163219, 0.0141305, 0.0163416, -0.0015827, 0.0016344
3: 0.0001125, 0.0016465, -0.0000014, 0.0016613, -0.0011846, 0.0012272
4: -0.0042759, -0.0028609, -0.0043809, -0.0028473, -0.0011356, 0.0011470
5: 0.0080505, 0.0095817, 0.0079368, 0.0095964, -0.0011819, 0.0012247
6: 0.0093171, 0.0098949, 0.0093115, 0.0099378, -0.0005090, 0.0005834
7: -0.0192002, -0.0158761, -0.0192322, -0.0156294, -0.0026379, 0.0025060
8: 0.9687800, 0.9783039, 0.9686883, 0.9790108, -0.0076348, 0.0074015
9: 0.0040296, 0.0068287, 0.0038219, 0.0068557, -0.0021289, 0.0022282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055278
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055278
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002761, 0.0000562, -0.0002939, 0.0000514, -0.0002386, 0.0002651
1: 0.0000278, 0.0014078, -0.0000558, 0.0014004, -0.0010094, 0.0011188
2: 0.0142316, 0.0162984, 0.0142427, 0.0164236, -0.0016702, 0.0015037
3: 0.0000747, 0.0016288, 0.0000830, 0.0017229, -0.0012535, 0.0011271
4: -0.0043107, -0.0028772, -0.0043031, -0.0027904, -0.0011750, 0.0010698
5: 0.0080127, 0.0095640, 0.0080210, 0.0096580, -0.0012510, 0.0011247
6: 0.0093237, 0.0099091, 0.0092883, 0.0099060, -0.0005242, 0.0005293
7: -0.0191618, -0.0157942, -0.0193658, -0.0158122, -0.0024016, 0.0026877
8: 0.9688900, 0.9785386, 0.9683055, 0.9784872, -0.0070281, 0.0078031
9: 0.0039607, 0.0067964, 0.0039758, 0.0069682, -0.0022723, 0.0020350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055009
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055259
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002761, 0.0000562, -0.0002871, 0.0000923, -0.0002803, 0.0002589
1: 0.0000278, 0.0014078, -0.0000240, 0.0014631, -0.0010868, 0.0011088
2: 0.0142316, 0.0162984, 0.0141489, 0.0163760, -0.0016499, 0.0016197
3: 0.0000747, 0.0016288, 0.0000125, 0.0016872, -0.0012358, 0.0012144
4: -0.0043107, -0.0028772, -0.0043681, -0.0028234, -0.0011800, 0.0011503
5: 0.0080127, 0.0095640, 0.0079506, 0.0096222, -0.0012331, 0.0012118
6: 0.0093237, 0.0099091, 0.0093017, 0.0099326, -0.0005571, 0.0006074
7: -0.0191618, -0.0157942, -0.0192883, -0.0156594, -0.0025907, 0.0026238
8: 0.9688900, 0.9785386, 0.9685276, 0.9789250, -0.0075698, 0.0077138
9: 0.0039607, 0.0067964, 0.0038471, 0.0069029, -0.0022267, 0.0021942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055009
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055259
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001046, -0.0002892, 0.0000509, -0.0002421, 0.0003062
1: 0.0000573, 0.0014819, -0.0000339, 0.0013996, -0.0010611, 0.0011658
2: 0.0141207, 0.0162542, 0.0142439, 0.0163907, -0.0017406, 0.0015704
3: -0.0000087, 0.0015956, 0.0000839, 0.0016982, -0.0013065, 0.0011727
4: -0.0043877, -0.0029079, -0.0043022, -0.0028132, -0.0012234, 0.0011524
5: 0.0079295, 0.0095308, 0.0080219, 0.0096333, -0.0013039, 0.0011698
6: 0.0093362, 0.0099405, 0.0092976, 0.0099057, -0.0005694, 0.0005495
7: -0.0190899, -0.0156135, -0.0193123, -0.0158142, -0.0024586, 0.0028036
8: 0.9690962, 0.9790565, 0.9684589, 0.9784814, -0.0073501, 0.0081315
9: 0.0038085, 0.0067358, 0.0039775, 0.0069231, -0.0023697, 0.0020951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0056726
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0056906
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0002698, 0.0001046, -0.0002821, 0.0000923, -0.0002742, 0.0002962
1: 0.0000573, 0.0014819, -0.0000004, 0.0014631, -0.0011020, 0.0011435
2: 0.0141207, 0.0162542, 0.0141488, 0.0163406, -0.0017008, 0.0016298
3: -0.0000087, 0.0015956, 0.0000124, 0.0016605, -0.0012738, 0.0012170
4: -0.0043877, -0.0029079, -0.0043682, -0.0028479, -0.0012180, 0.0012005
5: 0.0079295, 0.0095308, 0.0079506, 0.0095957, -0.0012709, 0.0012139
6: 0.0093362, 0.0099405, 0.0093118, 0.0099326, -0.0005963, 0.0006288
7: -0.0190899, -0.0156135, -0.0192306, -0.0156593, -0.0025469, 0.0027039
8: 0.9690962, 0.9790565, 0.9686929, 0.9789252, -0.0076299, 0.0079519
9: 0.0038085, 0.0067358, 0.0038470, 0.0068543, -0.0022948, 0.0021714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0057816
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0058219
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002697, 0.0001010, -0.0002939, 0.0000514, -0.0002435, 0.0003123
1: 0.0000577, 0.0014764, -0.0000558, 0.0014004, -0.0010644, 0.0012188
2: 0.0141290, 0.0162536, 0.0142427, 0.0164236, -0.0018199, 0.0015762
3: -0.0000025, 0.0015952, 0.0000830, 0.0017229, -0.0013660, 0.0011774
4: -0.0043819, -0.0029083, -0.0043031, -0.0027904, -0.0012788, 0.0011529
5: 0.0079357, 0.0095304, 0.0080210, 0.0096580, -0.0013633, 0.0011745
6: 0.0093364, 0.0099382, 0.0092883, 0.0099060, -0.0005696, 0.0005717
7: -0.0190889, -0.0156269, -0.0193658, -0.0158122, -0.0024729, 0.0029315
8: 0.9690988, 0.9790179, 0.9683055, 0.9784872, -0.0073764, 0.0085017
9: 0.0038198, 0.0067350, 0.0039758, 0.0069682, -0.0024776, 0.0021059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0056512
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0056906
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002697, 0.0001010, -0.0002871, 0.0000923, -0.0002752, 0.0002987
1: 0.0000577, 0.0014764, -0.0000240, 0.0014631, -0.0011047, 0.0011644
2: 0.0141290, 0.0162536, 0.0141489, 0.0163760, -0.0017334, 0.0016350
3: -0.0000025, 0.0015952, 0.0000125, 0.0016872, -0.0012986, 0.0012214
4: -0.0043819, -0.0029083, -0.0043681, -0.0028234, -0.0012377, 0.0012001
5: 0.0079357, 0.0095304, 0.0079506, 0.0096222, -0.0012958, 0.0012184
6: 0.0093364, 0.0099382, 0.0093017, 0.0099326, -0.0005962, 0.0006365
7: -0.0190889, -0.0156269, -0.0192883, -0.0156594, -0.0025607, 0.0027572
8: 0.9690988, 0.9790179, 0.9685276, 0.9789250, -0.0076522, 0.0081033
9: 0.0038198, 0.0067350, 0.0038471, 0.0069029, -0.0023403, 0.0021818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0057716
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0058219
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002822, 0.0001003, -0.0002757, 0.0000360, -0.0002361, 0.0002838
1: -0.0000011, 0.0014753, 0.0000295, 0.0013767, -0.0010615, 0.0010690
2: 0.0141305, 0.0163416, 0.0142782, 0.0162958, -0.0015963, 0.0015778
3: -0.0000014, 0.0016613, 0.0001097, 0.0016268, -0.0011982, 0.0011809
4: -0.0043809, -0.0028473, -0.0042785, -0.0028790, -0.0011212, 0.0011322
5: 0.0079368, 0.0095964, 0.0080477, 0.0095620, -0.0011958, 0.0011782
6: 0.0093115, 0.0099378, 0.0093245, 0.0098959, -0.0005844, 0.0005034
7: -0.0192322, -0.0156294, -0.0191576, -0.0158701, -0.0024979, 0.0025724
8: 0.9686883, 0.9790108, 0.9689021, 0.9783213, -0.0073785, 0.0074570
9: 0.0038219, 0.0068557, 0.0040245, 0.0067928, -0.0021736, 0.0021222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0053245
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0055508
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002822, 0.0001003, -0.0002794, 0.0000343, -0.0002362, 0.0002895
1: -0.0000011, 0.0014753, 0.0000120, 0.0013742, -0.0010648, 0.0010943
2: 0.0141305, 0.0163416, 0.0142819, 0.0163219, -0.0016344, 0.0015827
3: -0.0000014, 0.0016613, 0.0001125, 0.0016465, -0.0012272, 0.0011846
4: -0.0043809, -0.0028473, -0.0042759, -0.0028609, -0.0011470, 0.0011356
5: 0.0079368, 0.0095964, 0.0080505, 0.0095817, -0.0012247, 0.0011819
6: 0.0093115, 0.0099378, 0.0093171, 0.0098949, -0.0005834, 0.0005090
7: -0.0192322, -0.0156294, -0.0192002, -0.0158761, -0.0025060, 0.0026379
8: 0.9686883, 0.9790108, 0.9687800, 0.9783039, -0.0074015, 0.0076348
9: 0.0038219, 0.0068557, 0.0040296, 0.0068287, -0.0022282, 0.0021289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0053245
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0055508
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0002939, 0.0000514, -0.0002761, 0.0000562, -0.0002651, 0.0002386
1: -0.0000558, 0.0014004, 0.0000278, 0.0014078, -0.0011188, 0.0010094
2: 0.0142427, 0.0164236, 0.0142316, 0.0162984, -0.0015037, 0.0016702
3: 0.0000830, 0.0017229, 0.0000747, 0.0016288, -0.0011271, 0.0012535
4: -0.0043031, -0.0027904, -0.0043107, -0.0028772, -0.0010698, 0.0011750
5: 0.0080210, 0.0096580, 0.0080127, 0.0095640, -0.0011247, 0.0012510
6: 0.0092883, 0.0099060, 0.0093237, 0.0099091, -0.0005293, 0.0005242
7: -0.0193658, -0.0158122, -0.0191618, -0.0157942, -0.0026877, 0.0024016
8: 0.9683055, 0.9784872, 0.9688900, 0.9785386, -0.0078031, 0.0070281
9: 0.0039758, 0.0069682, 0.0039607, 0.0067964, -0.0020350, 0.0022723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000923, -0.0002761, 0.0000562, -0.0002589, 0.0002803
1: -0.0000240, 0.0014631, 0.0000278, 0.0014078, -0.0011088, 0.0010868
2: 0.0141489, 0.0163760, 0.0142316, 0.0162984, -0.0016197, 0.0016499
3: 0.0000125, 0.0016872, 0.0000747, 0.0016288, -0.0012144, 0.0012358
4: -0.0043681, -0.0028234, -0.0043107, -0.0028772, -0.0011503, 0.0011800
5: 0.0079506, 0.0096222, 0.0080127, 0.0095640, -0.0012118, 0.0012331
6: 0.0093017, 0.0099326, 0.0093237, 0.0099091, -0.0006074, 0.0005571
7: -0.0192883, -0.0156594, -0.0191618, -0.0157942, -0.0026238, 0.0025907
8: 0.9685276, 0.9789250, 0.9688900, 0.9785386, -0.0077138, 0.0075698
9: 0.0038471, 0.0069029, 0.0039607, 0.0067964, -0.0021942, 0.0022267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055656
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055656
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002698, 0.0001046, -0.0003062, 0.0002421
1: -0.0000339, 0.0013996, 0.0000573, 0.0014819, -0.0011658, 0.0010611
2: 0.0142439, 0.0163907, 0.0141207, 0.0162542, -0.0015704, 0.0017406
3: 0.0000839, 0.0016982, -0.0000087, 0.0015956, -0.0011727, 0.0013065
4: -0.0043022, -0.0028132, -0.0043877, -0.0029079, -0.0011524, 0.0012234
5: 0.0080219, 0.0096333, 0.0079295, 0.0095308, -0.0011698, 0.0013039
6: 0.0092976, 0.0099057, 0.0093362, 0.0099405, -0.0005495, 0.0005694
7: -0.0193123, -0.0158142, -0.0190899, -0.0156135, -0.0028036, 0.0024586
8: 0.9684589, 0.9784814, 0.9690962, 0.9790565, -0.0081315, 0.0073501
9: 0.0039775, 0.0069231, 0.0038085, 0.0067358, -0.0020951, 0.0023697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0053245
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0053245
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002821, 0.0000923, -0.0002698, 0.0001046, -0.0002962, 0.0002742
1: -0.0000004, 0.0014631, 0.0000573, 0.0014819, -0.0011435, 0.0011020
2: 0.0141488, 0.0163406, 0.0141207, 0.0162542, -0.0016298, 0.0017008
3: 0.0000124, 0.0016605, -0.0000087, 0.0015956, -0.0012170, 0.0012738
4: -0.0043682, -0.0028479, -0.0043877, -0.0029079, -0.0012005, 0.0012180
5: 0.0079506, 0.0095957, 0.0079295, 0.0095308, -0.0012139, 0.0012709
6: 0.0093118, 0.0099326, 0.0093362, 0.0099405, -0.0006288, 0.0005963
7: -0.0192306, -0.0156593, -0.0190899, -0.0156135, -0.0027039, 0.0025469
8: 0.9686929, 0.9789252, 0.9690962, 0.9790565, -0.0079519, 0.0076299
9: 0.0038470, 0.0068543, 0.0038085, 0.0067358, -0.0021714, 0.0022948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0056310
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0056310
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0002939, 0.0000514, -0.0002697, 0.0001010, -0.0003123, 0.0002435
1: -0.0000558, 0.0014004, 0.0000577, 0.0014764, -0.0012188, 0.0010644
2: 0.0142427, 0.0164236, 0.0141290, 0.0162536, -0.0015762, 0.0018199
3: 0.0000830, 0.0017229, -0.0000025, 0.0015952, -0.0011774, 0.0013660
4: -0.0043031, -0.0027904, -0.0043819, -0.0029083, -0.0011529, 0.0012788
5: 0.0080210, 0.0096580, 0.0079357, 0.0095304, -0.0011745, 0.0013633
6: 0.0092883, 0.0099060, 0.0093364, 0.0099382, -0.0005717, 0.0005696
7: -0.0193658, -0.0158122, -0.0190889, -0.0156269, -0.0029315, 0.0024729
8: 0.9683055, 0.9784872, 0.9690988, 0.9790179, -0.0085017, 0.0073764
9: 0.0039758, 0.0069682, 0.0038198, 0.0067350, -0.0021059, 0.0024776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000923, -0.0002697, 0.0001010, -0.0002987, 0.0002752
1: -0.0000240, 0.0014631, 0.0000577, 0.0014764, -0.0011644, 0.0011047
2: 0.0141489, 0.0163760, 0.0141290, 0.0162536, -0.0016350, 0.0017334
3: 0.0000125, 0.0016872, -0.0000025, 0.0015952, -0.0012214, 0.0012986
4: -0.0043681, -0.0028234, -0.0043819, -0.0029083, -0.0012001, 0.0012377
5: 0.0079506, 0.0096222, 0.0079357, 0.0095304, -0.0012184, 0.0012958
6: 0.0093017, 0.0099326, 0.0093364, 0.0099382, -0.0006365, 0.0005962
7: -0.0192883, -0.0156594, -0.0190889, -0.0156269, -0.0027572, 0.0025607
8: 0.9685276, 0.9789250, 0.9690988, 0.9790179, -0.0081033, 0.0076522
9: 0.0038471, 0.0069029, 0.0038198, 0.0067350, -0.0021818, 0.0023403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056607
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002822, 0.0001003, -0.0002796, 0.0002334
1: -0.0000339, 0.0013996, -0.0000011, 0.0014753, -0.0010609, 0.0010260
2: 0.0142439, 0.0163907, 0.0141305, 0.0163416, -0.0015187, 0.0015813
3: 0.0000839, 0.0016982, -0.0000014, 0.0016613, -0.0011341, 0.0011857
4: -0.0043022, -0.0028132, -0.0043809, -0.0028473, -0.0011154, 0.0011220
5: 0.0080219, 0.0096333, 0.0079368, 0.0095964, -0.0011313, 0.0011832
6: 0.0092976, 0.0099057, 0.0093115, 0.0099378, -0.0005370, 0.0005942
7: -0.0193123, -0.0158142, -0.0192322, -0.0156294, -0.0025299, 0.0023735
8: 0.9684589, 0.9784814, 0.9686883, 0.9790108, -0.0073900, 0.0071082
9: 0.0039775, 0.0069231, 0.0038219, 0.0068557, -0.0020238, 0.0021429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053368
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053368
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002935, 0.0000514, -0.0002822, 0.0001003, -0.0002854, 0.0002352
1: -0.0000536, 0.0014004, -0.0000011, 0.0014753, -0.0010868, 0.0010298
2: 0.0142427, 0.0164203, 0.0141305, 0.0163416, -0.0015245, 0.0016201
3: 0.0000830, 0.0017204, -0.0000014, 0.0016613, -0.0011384, 0.0012149
4: -0.0043031, -0.0027927, -0.0043809, -0.0028473, -0.0011194, 0.0011487
5: 0.0080210, 0.0096555, 0.0079368, 0.0095964, -0.0011356, 0.0012123
6: 0.0092892, 0.0099060, 0.0093115, 0.0099378, -0.0005454, 0.0005945
7: -0.0193604, -0.0158122, -0.0192322, -0.0156294, -0.0025958, 0.0023829
8: 0.9683210, 0.9784871, 0.9686883, 0.9790108, -0.0075713, 0.0071350
9: 0.0039758, 0.0069636, 0.0038219, 0.0068557, -0.0020317, 0.0021972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053482
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053482
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002897, 0.0000723, -0.0002939, 0.0000514, -0.0002324, 0.0002601
1: -0.0000359, 0.0014323, -0.0000558, 0.0014004, -0.0009960, 0.0010565
2: 0.0141949, 0.0163937, 0.0142427, 0.0164236, -0.0015741, 0.0014789
3: 0.0000471, 0.0017005, 0.0000830, 0.0017229, -0.0011798, 0.0011066
4: -0.0043362, -0.0028111, -0.0043031, -0.0027904, -0.0011192, 0.0010673
5: 0.0079851, 0.0096355, 0.0080210, 0.0096580, -0.0011773, 0.0011040
6: 0.0092967, 0.0099195, 0.0092883, 0.0099060, -0.0005622, 0.0005354
7: -0.0193172, -0.0157343, -0.0193658, -0.0158122, -0.0023384, 0.0025171
8: 0.9684449, 0.9787102, 0.9683055, 0.9784872, -0.0069172, 0.0073571
9: 0.0039102, 0.0069272, 0.0039758, 0.0069682, -0.0021320, 0.0019880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053245
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002897, 0.0000723, -0.0002871, 0.0000923, -0.0002782, 0.0002588
1: -0.0000359, 0.0014323, -0.0000240, 0.0014631, -0.0010956, 0.0010780
2: 0.0141949, 0.0163937, 0.0141489, 0.0163760, -0.0015983, 0.0016282
3: 0.0000471, 0.0017005, 0.0000125, 0.0016872, -0.0011944, 0.0012188
4: -0.0043362, -0.0028111, -0.0043681, -0.0028234, -0.0011607, 0.0011708
5: 0.0079851, 0.0096355, 0.0079506, 0.0096222, -0.0011916, 0.0012161
6: 0.0092967, 0.0099195, 0.0093017, 0.0099326, -0.0006045, 0.0006178
7: -0.0193172, -0.0157343, -0.0192883, -0.0156594, -0.0025816, 0.0025100
8: 0.9684449, 0.9787102, 0.9685276, 0.9789250, -0.0076140, 0.0074786
9: 0.0039102, 0.0069272, 0.0038471, 0.0069029, -0.0021364, 0.0021927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053245
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0002826, 0.0001164, -0.0002892, 0.0000509, -0.0002348, 0.0002989
1: -0.0000029, 0.0015000, -0.0000339, 0.0013996, -0.0010490, 0.0011049
2: 0.0140936, 0.0163443, 0.0142439, 0.0163907, -0.0016471, 0.0015470
3: -0.0000291, 0.0016633, 0.0000839, 0.0016982, -0.0012352, 0.0011530
4: -0.0044065, -0.0028454, -0.0043022, -0.0028132, -0.0011676, 0.0011509
5: 0.0079091, 0.0095984, 0.0080219, 0.0096333, -0.0012326, 0.0011500
6: 0.0093107, 0.0099482, 0.0092976, 0.0099057, -0.0005949, 0.0005556
7: -0.0192366, -0.0155693, -0.0193123, -0.0158142, -0.0023940, 0.0026371
8: 0.9686757, 0.9791829, 0.9684589, 0.9784814, -0.0072470, 0.0076972
9: 0.0037713, 0.0068594, 0.0039775, 0.0069231, -0.0022332, 0.0020472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055520
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055623
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0002826, 0.0001164, -0.0002821, 0.0000923, -0.0002686, 0.0002897
1: -0.0000029, 0.0015000, -0.0000004, 0.0014631, -0.0010917, 0.0010971
2: 0.0140936, 0.0163443, 0.0141488, 0.0163406, -0.0016247, 0.0016080
3: -0.0000291, 0.0016633, 0.0000124, 0.0016605, -0.0012139, 0.0011977
4: -0.0044065, -0.0028454, -0.0043682, -0.0028479, -0.0011897, 0.0012026
5: 0.0079091, 0.0095984, 0.0079506, 0.0095957, -0.0012110, 0.0011945
6: 0.0093107, 0.0099482, 0.0093118, 0.0099326, -0.0006219, 0.0006364
7: -0.0192366, -0.0155693, -0.0192306, -0.0156593, -0.0024851, 0.0025495
8: 0.9686757, 0.9791829, 0.9686929, 0.9789252, -0.0075343, 0.0076033
9: 0.0037713, 0.0068594, 0.0038470, 0.0068543, -0.0021721, 0.0021260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056340
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056601
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002825, 0.0001129, -0.0002939, 0.0000514, -0.0002369, 0.0003047
1: -0.0000024, 0.0014946, -0.0000558, 0.0014004, -0.0010560, 0.0011572
2: 0.0141016, 0.0163436, 0.0142427, 0.0164236, -0.0017248, 0.0015585
3: -0.0000231, 0.0016628, 0.0000830, 0.0017229, -0.0012932, 0.0011621
4: -0.0044009, -0.0028459, -0.0043031, -0.0027904, -0.0012238, 0.0011556
5: 0.0079151, 0.0095979, 0.0080210, 0.0096580, -0.0012904, 0.0011591
6: 0.0093109, 0.0099460, 0.0092883, 0.0099060, -0.0005951, 0.0005780
7: -0.0192355, -0.0155824, -0.0193658, -0.0158122, -0.0024179, 0.0027627
8: 0.9686789, 0.9791455, 0.9683055, 0.9784872, -0.0072998, 0.0080607
9: 0.0037823, 0.0068584, 0.0039758, 0.0069682, -0.0023388, 0.0020660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055486
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055623
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002825, 0.0001129, -0.0002871, 0.0000923, -0.0002702, 0.0002918
1: -0.0000024, 0.0014946, -0.0000240, 0.0014631, -0.0010981, 0.0011189
2: 0.0141016, 0.0163436, 0.0141489, 0.0163760, -0.0016578, 0.0016186
3: -0.0000231, 0.0016628, 0.0000125, 0.0016872, -0.0012385, 0.0012062
4: -0.0044009, -0.0028459, -0.0043681, -0.0028234, -0.0012081, 0.0012066
5: 0.0079151, 0.0095979, 0.0079506, 0.0096222, -0.0012355, 0.0012030
6: 0.0093109, 0.0099460, 0.0093017, 0.0099326, -0.0006216, 0.0006442
7: -0.0192355, -0.0155824, -0.0192883, -0.0156594, -0.0025086, 0.0025976
8: 0.9686789, 0.9791455, 0.9685276, 0.9789250, -0.0075829, 0.0077578
9: 0.0037823, 0.0068584, 0.0038471, 0.0069029, -0.0022126, 0.0021443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056297
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056601
time: 0.68 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.62 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053413
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053413
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053506
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053506
IS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053265
IS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053504
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053265
IS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0053504
IS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0055550
IS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0055656
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0056347
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0056607
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0055508
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0055656
IS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0056310
IS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053265, upper bound: 0.0056607
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055174
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055174
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055278
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055278
IS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055009
IS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055259
IS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055009
IS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0055259
IS_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0056726
IS_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0056906
IS_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0057816
IS_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0058219
IS_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0056512
IS_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0056906
IS_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0057716
IS_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0053245, upper bound: 0.0058219
IS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0053245
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0055508
IS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0053245
IS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0055508
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
IS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055656
IS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055656
IS_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0053245
IS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0053245
IS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0056310
IS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055174, upper bound: 0.0056310
IS_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
IS_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
IS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
IS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056607
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053368
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053368
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053482
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053482
IS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053245
IS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
IS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053245
IS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0053479
IS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055520
IS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055623
IS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056340
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056601
IS_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055486
IS_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0055623
IS_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056297
IS_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 8, lower bound: -0.0055009, upper bound: 0.0056601

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002757, 0.0000360, -0.0002095, 0.0002095
1: 0.0000295, 0.0013767, 0.0000295, 0.0013767, -0.0009052, 0.0009052
2: 0.0142782, 0.0162958, 0.0142782, 0.0162958, -0.0013510, 0.0013510
3: 0.0001097, 0.0016268, 0.0001097, 0.0016268, -0.0010137, 0.0010137
4: -0.0042785, -0.0028790, -0.0042785, -0.0028790, -0.0009511, 0.0009511
5: 0.0080477, 0.0095620, 0.0080477, 0.0095620, -0.0010116, 0.0010116
6: 0.0093245, 0.0098959, 0.0093245, 0.0098959, -0.0004340, 0.0004340
7: -0.0191576, -0.0158701, -0.0191576, -0.0158701, -0.0021727, 0.0021727
8: 0.9689021, 0.9783213, 0.9689021, 0.9783213, -0.0063118, 0.0063118
9: 0.0040245, 0.0067928, 0.0040245, 0.0067928, -0.0018370, 0.0018370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037662, upper bound: 0.0034613
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053035, upper bound: 0.0053035
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002693, 0.0000816, -0.0002582, 0.0002132
1: 0.0000295, 0.0013767, 0.0000593, 0.0014467, -0.0010044, 0.0009526
2: 0.0142782, 0.0162958, 0.0141734, 0.0162511, -0.0014140, 0.0014996
3: 0.0001097, 0.0016268, 0.0000309, 0.0015933, -0.0010578, 0.0011254
4: -0.0042785, -0.0028790, -0.0043511, -0.0029100, -0.0010234, 0.0010541
5: 0.0080477, 0.0095620, 0.0079690, 0.0095285, -0.0010553, 0.0011232
6: 0.0093245, 0.0098959, 0.0093371, 0.0099256, -0.0004761, 0.0005559
7: -0.0191576, -0.0158701, -0.0190849, -0.0156993, -0.0024148, 0.0022352
8: 0.9689021, 0.9783213, 0.9691105, 0.9788105, -0.0070056, 0.0066139
9: 0.0040245, 0.0067928, 0.0038807, 0.0067316, -0.0019000, 0.0020409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034613, upper bound: 0.0042326
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053035, upper bound: 0.0053035
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002757, 0.0000360, -0.0002153, 0.0002100
1: 0.0000120, 0.0013742, 0.0000295, 0.0013767, -0.0009305, 0.0009123
2: 0.0142819, 0.0163219, 0.0142782, 0.0162958, -0.0013616, 0.0013891
3: 0.0001125, 0.0016465, 0.0001097, 0.0016268, -0.0010217, 0.0010427
4: -0.0042759, -0.0028609, -0.0042785, -0.0028790, -0.0009584, 0.0009768
5: 0.0080505, 0.0095817, 0.0080477, 0.0095620, -0.0010196, 0.0010406
6: 0.0093171, 0.0098949, 0.0093245, 0.0098959, -0.0004395, 0.0004370
7: -0.0192002, -0.0158761, -0.0191576, -0.0158701, -0.0022382, 0.0021900
8: 0.9687800, 0.9783039, 0.9689021, 0.9783213, -0.0064897, 0.0063614
9: 0.0040296, 0.0068287, 0.0040245, 0.0067928, -0.0018516, 0.0018916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032678, upper bound: 0.0035494
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0052858
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002693, 0.0000816, -0.0002640, 0.0002133
1: 0.0000120, 0.0013742, 0.0000593, 0.0014467, -0.0010297, 0.0009559
2: 0.0142819, 0.0163219, 0.0141734, 0.0162511, -0.0014189, 0.0015377
3: 0.0001125, 0.0016465, 0.0000309, 0.0015933, -0.0010615, 0.0011544
4: -0.0042759, -0.0028609, -0.0043511, -0.0029100, -0.0010268, 0.0010799
5: 0.0080505, 0.0095817, 0.0079690, 0.0095285, -0.0010590, 0.0011522
6: 0.0093171, 0.0098949, 0.0093371, 0.0099256, -0.0004816, 0.0005573
7: -0.0192002, -0.0158761, -0.0190849, -0.0156993, -0.0024803, 0.0022432
8: 0.9687800, 0.9783039, 0.9691105, 0.9788105, -0.0071834, 0.0066369
9: 0.0040296, 0.0068287, 0.0038807, 0.0067316, -0.0019068, 0.0020955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032678, upper bound: 0.0040848
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0052858
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002797, 0.0000343, -0.0002100, 0.0002200
1: 0.0000295, 0.0013767, 0.0000107, 0.0013742, -0.0009123, 0.0009586
2: 0.0142782, 0.0162958, 0.0142819, 0.0163240, -0.0014311, 0.0013616
3: 0.0001097, 0.0016268, 0.0001125, 0.0016480, -0.0010741, 0.0010217
4: -0.0042785, -0.0028790, -0.0042759, -0.0028595, -0.0010066, 0.0009584
5: 0.0080477, 0.0095620, 0.0080505, 0.0095832, -0.0010720, 0.0010196
6: 0.0093245, 0.0098959, 0.0093165, 0.0098949, -0.0004370, 0.0004545
7: -0.0191576, -0.0158701, -0.0192035, -0.0158761, -0.0021900, 0.0023044
8: 0.9689021, 0.9783213, 0.9687706, 0.9783039, -0.0063614, 0.0066857
9: 0.0040245, 0.0067928, 0.0040296, 0.0068315, -0.0019478, 0.0018516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052348, upper bound: 0.0052341
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052348, upper bound: 0.0052348
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002797, 0.0000343, -0.0002097, 0.0002155
1: 0.0000120, 0.0013742, 0.0000107, 0.0013742, -0.0009095, 0.0009445
2: 0.0142819, 0.0163219, 0.0142819, 0.0163240, -0.0014096, 0.0013576
3: 0.0001125, 0.0016465, 0.0001125, 0.0016480, -0.0010579, 0.0010189
4: -0.0042759, -0.0028609, -0.0042759, -0.0028595, -0.0009927, 0.0009551
5: 0.0080505, 0.0095817, 0.0080505, 0.0095832, -0.0010557, 0.0010169
6: 0.0093171, 0.0098949, 0.0093165, 0.0098949, -0.0004316, 0.0004487
7: -0.0192002, -0.0158761, -0.0192035, -0.0158761, -0.0021852, 0.0022673
8: 0.9687800, 0.9783039, 0.9687706, 0.9783039, -0.0063427, 0.0065857
9: 0.0040296, 0.0068287, 0.0040296, 0.0068315, -0.0019173, 0.0018474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052348, upper bound: 0.0052572
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052348, upper bound: 0.0052553
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002734, 0.0000794, -0.0002570, 0.0002189
1: 0.0000295, 0.0013767, 0.0000401, 0.0014433, -0.0010091, 0.0009798
2: 0.0142782, 0.0162958, 0.0141785, 0.0162800, -0.0014545, 0.0015065
3: 0.0001097, 0.0016268, 0.0000347, 0.0016150, -0.0010881, 0.0011307
4: -0.0042785, -0.0028790, -0.0043476, -0.0028900, -0.0010486, 0.0010590
5: 0.0080477, 0.0095620, 0.0079728, 0.0095502, -0.0010856, 0.0011284
6: 0.0093245, 0.0098959, 0.0093289, 0.0099242, -0.0004780, 0.0005556
7: -0.0191576, -0.0158701, -0.0191318, -0.0157076, -0.0024262, 0.0023004
8: 0.9689021, 0.9783213, 0.9689760, 0.9787869, -0.0070381, 0.0068038
9: 0.0040245, 0.0067928, 0.0038877, 0.0067711, -0.0019547, 0.0020505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044759, upper bound: 0.0040537
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054820, upper bound: 0.0052597
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002734, 0.0000794, -0.0002578, 0.0002139
1: 0.0000120, 0.0013742, 0.0000401, 0.0014433, -0.0010092, 0.0009617
2: 0.0142819, 0.0163219, 0.0141785, 0.0162800, -0.0014277, 0.0015070
3: 0.0001125, 0.0016465, 0.0000347, 0.0016150, -0.0010679, 0.0011312
4: -0.0042759, -0.0028609, -0.0043476, -0.0028900, -0.0010328, 0.0010587
5: 0.0080505, 0.0095817, 0.0079728, 0.0095502, -0.0010655, 0.0011290
6: 0.0093171, 0.0098949, 0.0093289, 0.0099242, -0.0004739, 0.0005529
7: -0.0192002, -0.0158761, -0.0191318, -0.0157076, -0.0024285, 0.0022570
8: 0.9687800, 0.9783039, 0.9689760, 0.9787869, -0.0070399, 0.0066779
9: 0.0040296, 0.0068287, 0.0038877, 0.0067711, -0.0019178, 0.0020523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044759, upper bound: 0.0040537
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054820, upper bound: 0.0052858
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000816, -0.0002757, 0.0000360, -0.0002132, 0.0002582
1: 0.0000593, 0.0014467, 0.0000295, 0.0013767, -0.0009526, 0.0010044
2: 0.0141734, 0.0162511, 0.0142782, 0.0162958, -0.0014996, 0.0014140
3: 0.0000309, 0.0015933, 0.0001097, 0.0016268, -0.0011254, 0.0010578
4: -0.0043511, -0.0029100, -0.0042785, -0.0028790, -0.0010541, 0.0010234
5: 0.0079690, 0.0095285, 0.0080477, 0.0095620, -0.0011232, 0.0010553
6: 0.0093371, 0.0099256, 0.0093245, 0.0098959, -0.0005559, 0.0004761
7: -0.0190849, -0.0156993, -0.0191576, -0.0158701, -0.0022352, 0.0024148
8: 0.9691105, 0.9788105, 0.9689021, 0.9783213, -0.0066139, 0.0070056
9: 0.0038807, 0.0067316, 0.0040245, 0.0067928, -0.0020409, 0.0019000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0040984, upper bound: 0.0045137
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0054864
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000794, -0.0002757, 0.0000360, -0.0002189, 0.0002570
1: 0.0000401, 0.0014433, 0.0000295, 0.0013767, -0.0009798, 0.0010091
2: 0.0141785, 0.0162800, 0.0142782, 0.0162958, -0.0015065, 0.0014545
3: 0.0000347, 0.0016150, 0.0001097, 0.0016268, -0.0011307, 0.0010881
4: -0.0043476, -0.0028900, -0.0042785, -0.0028790, -0.0010590, 0.0010486
5: 0.0079728, 0.0095502, 0.0080477, 0.0095620, -0.0011284, 0.0010856
6: 0.0093289, 0.0099242, 0.0093245, 0.0098959, -0.0005556, 0.0004780
7: -0.0191318, -0.0157076, -0.0191576, -0.0158701, -0.0023004, 0.0024262
8: 0.9689760, 0.9787869, 0.9689021, 0.9783213, -0.0068038, 0.0070381
9: 0.0038877, 0.0067711, 0.0040245, 0.0067928, -0.0020505, 0.0019547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0040984, upper bound: 0.0045137
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0054989
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000816, -0.0002693, 0.0000816, -0.0002490, 0.0002490
1: 0.0000593, 0.0014467, 0.0000593, 0.0014467, -0.0009929, 0.0009929
2: 0.0141734, 0.0162511, 0.0141734, 0.0162511, -0.0014732, 0.0014732
3: 0.0000309, 0.0015933, 0.0000309, 0.0015933, -0.0011016, 0.0011016
4: -0.0043511, -0.0029100, -0.0043511, -0.0029100, -0.0010697, 0.0010697
5: 0.0079690, 0.0095285, 0.0079690, 0.0095285, -0.0010990, 0.0010990
6: 0.0093371, 0.0099256, 0.0093371, 0.0099256, -0.0005840, 0.0005840
7: -0.0190849, -0.0156993, -0.0190849, -0.0156993, -0.0023234, 0.0023234
8: 0.9691105, 0.9788105, 0.9691105, 0.9788105, -0.0068920, 0.0068920
9: 0.0038807, 0.0067316, 0.0038807, 0.0067316, -0.0019760, 0.0019760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0050030
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054654, upper bound: 0.0055654
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000794, -0.0002693, 0.0000816, -0.0002544, 0.0002480
1: 0.0000401, 0.0014433, 0.0000593, 0.0014467, -0.0010181, 0.0009954
2: 0.0141785, 0.0162800, 0.0141734, 0.0162511, -0.0014769, 0.0015104
3: 0.0000347, 0.0016150, 0.0000309, 0.0015933, -0.0011044, 0.0011295
4: -0.0043476, -0.0028900, -0.0043511, -0.0029100, -0.0010722, 0.0010935
5: 0.0079728, 0.0095502, 0.0079690, 0.0095285, -0.0011018, 0.0011269
6: 0.0093289, 0.0099242, 0.0093371, 0.0099256, -0.0005829, 0.0005850
7: -0.0191318, -0.0157076, -0.0190849, -0.0156993, -0.0023856, 0.0023294
8: 0.9689760, 0.9787869, 0.9691105, 0.9788105, -0.0070656, 0.0069093
9: 0.0038877, 0.0067711, 0.0038807, 0.0067316, -0.0019810, 0.0020283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049112, upper bound: 0.0051199
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054654, upper bound: 0.0055927
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000816, -0.0002797, 0.0000343, -0.0002133, 0.0002681
1: 0.0000593, 0.0014467, 0.0000107, 0.0013742, -0.0009559, 0.0010573
2: 0.0141734, 0.0162511, 0.0142819, 0.0163240, -0.0015790, 0.0014189
3: 0.0000309, 0.0015933, 0.0001125, 0.0016480, -0.0011853, 0.0010615
4: -0.0043511, -0.0029100, -0.0042759, -0.0028595, -0.0011092, 0.0010268
5: 0.0079690, 0.0095285, 0.0080505, 0.0095832, -0.0011829, 0.0010590
6: 0.0093371, 0.0099256, 0.0093165, 0.0098949, -0.0005573, 0.0004964
7: -0.0190849, -0.0156993, -0.0192035, -0.0158761, -0.0022432, 0.0025453
8: 0.9691105, 0.9788105, 0.9687706, 0.9783039, -0.0066369, 0.0073761
9: 0.0038807, 0.0067316, 0.0040296, 0.0068315, -0.0021507, 0.0019068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0040537, upper bound: 0.0044759
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0054820
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000794, -0.0002797, 0.0000343, -0.0002139, 0.0002631
1: 0.0000401, 0.0014433, 0.0000107, 0.0013742, -0.0009617, 0.0010424
2: 0.0141785, 0.0162800, 0.0142819, 0.0163240, -0.0015563, 0.0014277
3: 0.0000347, 0.0016150, 0.0001125, 0.0016480, -0.0011681, 0.0010679
4: -0.0043476, -0.0028900, -0.0042759, -0.0028595, -0.0010944, 0.0010328
5: 0.0079728, 0.0095502, 0.0080505, 0.0095832, -0.0011658, 0.0010655
6: 0.0093289, 0.0099242, 0.0093165, 0.0098949, -0.0005529, 0.0004903
7: -0.0191318, -0.0157076, -0.0192035, -0.0158761, -0.0022570, 0.0025062
8: 0.9689760, 0.9787869, 0.9687706, 0.9783039, -0.0066779, 0.0072703
9: 0.0038877, 0.0067711, 0.0040296, 0.0068315, -0.0021185, 0.0019178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0040537, upper bound: 0.0044759
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0054989
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000816, -0.0002734, 0.0000794, -0.0002480, 0.0002544
1: 0.0000593, 0.0014467, 0.0000401, 0.0014433, -0.0009954, 0.0010181
2: 0.0141734, 0.0162511, 0.0141785, 0.0162800, -0.0015104, 0.0014769
3: 0.0000309, 0.0015933, 0.0000347, 0.0016150, -0.0011295, 0.0011044
4: -0.0043511, -0.0029100, -0.0043476, -0.0028900, -0.0010935, 0.0010722
5: 0.0079690, 0.0095285, 0.0079728, 0.0095502, -0.0011269, 0.0011018
6: 0.0093371, 0.0099256, 0.0093289, 0.0099242, -0.0005850, 0.0005829
7: -0.0190849, -0.0156993, -0.0191318, -0.0157076, -0.0023295, 0.0023856
8: 0.9691105, 0.9788105, 0.9689760, 0.9787869, -0.0069093, 0.0070656
9: 0.0038807, 0.0067316, 0.0038877, 0.0067711, -0.0020283, 0.0019810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0049914
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054654, upper bound: 0.0055614
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000794, -0.0002734, 0.0000794, -0.0002493, 0.0002493
1: 0.0000401, 0.0014433, 0.0000401, 0.0014433, -0.0010021, 0.0010021
2: 0.0141785, 0.0162800, 0.0141785, 0.0162800, -0.0014878, 0.0014878
3: 0.0000347, 0.0016150, 0.0000347, 0.0016150, -0.0011129, 0.0011129
4: -0.0043476, -0.0028900, -0.0043476, -0.0028900, -0.0010766, 0.0010766
5: 0.0079728, 0.0095502, 0.0079728, 0.0095502, -0.0011104, 0.0011104
6: 0.0093289, 0.0099242, 0.0093289, 0.0099242, -0.0005807, 0.0005807
7: -0.0191318, -0.0157076, -0.0191318, -0.0157076, -0.0023473, 0.0023473
8: 0.9689760, 0.9787869, 0.9689760, 0.9787869, -0.0069591, 0.0069591
9: 0.0038877, 0.0067711, 0.0038877, 0.0067711, -0.0019960, 0.0019960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0049917
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054654, upper bound: 0.0055927
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002892, 0.0000509, -0.0002367, 0.0002356
1: 0.0000295, 0.0013767, -0.0000339, 0.0013996, -0.0009888, 0.0010299
2: 0.0142782, 0.0162958, 0.0142439, 0.0163907, -0.0015372, 0.0014762
3: 0.0001097, 0.0016268, 0.0000839, 0.0016982, -0.0011535, 0.0011079
4: -0.0042785, -0.0028790, -0.0043022, -0.0028132, -0.0010823, 0.0010379
5: 0.0080477, 0.0095620, 0.0080219, 0.0096333, -0.0011512, 0.0011056
6: 0.0093245, 0.0098959, 0.0092976, 0.0099057, -0.0004694, 0.0004919
7: -0.0191576, -0.0158701, -0.0193123, -0.0158142, -0.0023767, 0.0024721
8: 0.9689021, 0.9783213, 0.9684589, 0.9784814, -0.0068964, 0.0071816
9: 0.0040245, 0.0067928, 0.0039775, 0.0069231, -0.0020906, 0.0020088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037568, upper bound: 0.0036264
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053000, upper bound: 0.0054778
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002821, 0.0000923, -0.0002783, 0.0002359
1: 0.0000295, 0.0013767, -0.0000004, 0.0014631, -0.0010656, 0.0010571
2: 0.0142782, 0.0162958, 0.0141488, 0.0163406, -0.0015720, 0.0015912
3: 0.0001097, 0.0016268, 0.0000124, 0.0016605, -0.0011770, 0.0011943
4: -0.0042785, -0.0028790, -0.0043682, -0.0028479, -0.0011248, 0.0011177
5: 0.0080477, 0.0095620, 0.0079506, 0.0095957, -0.0011744, 0.0011920
6: 0.0093245, 0.0098959, 0.0093118, 0.0099326, -0.0005020, 0.0005842
7: -0.0191576, -0.0158701, -0.0192306, -0.0156593, -0.0025641, 0.0024949
8: 0.9689021, 0.9783213, 0.9686929, 0.9789252, -0.0074334, 0.0073506
9: 0.0040245, 0.0067928, 0.0038470, 0.0068543, -0.0021180, 0.0021667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034750, upper bound: 0.0045079
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053000, upper bound: 0.0054778
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002892, 0.0000509, -0.0002425, 0.0002360
1: 0.0000120, 0.0013742, -0.0000339, 0.0013996, -0.0010141, 0.0010370
2: 0.0142819, 0.0163219, 0.0142439, 0.0163907, -0.0015478, 0.0015143
3: 0.0001125, 0.0016465, 0.0000839, 0.0016982, -0.0011615, 0.0011369
4: -0.0042759, -0.0028609, -0.0043022, -0.0028132, -0.0010897, 0.0010637
5: 0.0080505, 0.0095817, 0.0080219, 0.0096333, -0.0011592, 0.0011346
6: 0.0093171, 0.0098949, 0.0092976, 0.0099057, -0.0004750, 0.0004949
7: -0.0192002, -0.0158761, -0.0193123, -0.0158142, -0.0024422, 0.0024894
8: 0.9687800, 0.9783039, 0.9684589, 0.9784814, -0.0070742, 0.0072312
9: 0.0040296, 0.0068287, 0.0039775, 0.0069231, -0.0021051, 0.0020634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032842, upper bound: 0.0039565
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0054624
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002821, 0.0000923, -0.0002841, 0.0002360
1: 0.0000120, 0.0013742, -0.0000004, 0.0014631, -0.0010909, 0.0010604
2: 0.0142819, 0.0163219, 0.0141488, 0.0163406, -0.0015770, 0.0016294
3: 0.0001125, 0.0016465, 0.0000124, 0.0016605, -0.0011807, 0.0012233
4: -0.0042759, -0.0028609, -0.0043682, -0.0028479, -0.0011282, 0.0011434
5: 0.0080505, 0.0095817, 0.0079506, 0.0095957, -0.0011781, 0.0012209
6: 0.0093171, 0.0098949, 0.0093118, 0.0099326, -0.0005076, 0.0005831
7: -0.0192002, -0.0158761, -0.0192306, -0.0156593, -0.0026296, 0.0025029
8: 0.9687800, 0.9783039, 0.9686929, 0.9789252, -0.0076112, 0.0073736
9: 0.0040296, 0.0068287, 0.0038470, 0.0068543, -0.0021248, 0.0022212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032842, upper bound: 0.0043621
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0054624
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002939, 0.0000514, -0.0002383, 0.0002466
1: 0.0000295, 0.0013767, -0.0000558, 0.0014004, -0.0009953, 0.0010869
2: 0.0142782, 0.0162958, 0.0142427, 0.0164236, -0.0016224, 0.0014860
3: 0.0001097, 0.0016268, 0.0000830, 0.0017229, -0.0012175, 0.0011152
4: -0.0042785, -0.0028790, -0.0043031, -0.0027904, -0.0011418, 0.0010447
5: 0.0080477, 0.0095620, 0.0080210, 0.0096580, -0.0012150, 0.0011130
6: 0.0093245, 0.0098959, 0.0092883, 0.0099060, -0.0004722, 0.0005158
7: -0.0191576, -0.0158701, -0.0193658, -0.0158122, -0.0023927, 0.0026096
8: 0.9689021, 0.9783213, 0.9683055, 0.9784872, -0.0069421, 0.0075796
9: 0.0040245, 0.0067928, 0.0039758, 0.0069682, -0.0022065, 0.0020223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0053819
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0054277
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002939, 0.0000514, -0.0002379, 0.0002411
1: 0.0000120, 0.0013742, -0.0000558, 0.0014004, -0.0009927, 0.0010678
2: 0.0142819, 0.0163219, 0.0142427, 0.0164236, -0.0015938, 0.0014823
3: 0.0001125, 0.0016465, 0.0000830, 0.0017229, -0.0011960, 0.0011127
4: -0.0042759, -0.0028609, -0.0043031, -0.0027904, -0.0011221, 0.0010416
5: 0.0080505, 0.0095817, 0.0080210, 0.0096580, -0.0011936, 0.0011105
6: 0.0093171, 0.0098949, 0.0092883, 0.0099060, -0.0004669, 0.0005121
7: -0.0192002, -0.0158761, -0.0193658, -0.0158122, -0.0023884, 0.0025618
8: 0.9687800, 0.9783039, 0.9683055, 0.9784872, -0.0069248, 0.0074461
9: 0.0040296, 0.0068287, 0.0039758, 0.0069682, -0.0021669, 0.0020185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0054030
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0054561
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002871, 0.0000923, -0.0002796, 0.0002404
1: 0.0000295, 0.0013767, -0.0000240, 0.0014631, -0.0010684, 0.0010768
2: 0.0142782, 0.0162958, 0.0141489, 0.0163760, -0.0016021, 0.0015954
3: 0.0001097, 0.0016268, 0.0000125, 0.0016872, -0.0011998, 0.0011976
4: -0.0042785, -0.0028790, -0.0043681, -0.0028234, -0.0011468, 0.0011206
5: 0.0080477, 0.0095620, 0.0079506, 0.0096222, -0.0011971, 0.0011952
6: 0.0093245, 0.0098959, 0.0093017, 0.0099326, -0.0005032, 0.0005942
7: -0.0191576, -0.0158701, -0.0192883, -0.0156594, -0.0025711, 0.0025458
8: 0.9689021, 0.9783213, 0.9685276, 0.9789250, -0.0074533, 0.0074902
9: 0.0040245, 0.0067928, 0.0038471, 0.0069029, -0.0021609, 0.0021725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044759, upper bound: 0.0043340
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054820, upper bound: 0.0054362
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002871, 0.0000923, -0.0002796, 0.0002368
1: 0.0000120, 0.0013742, -0.0000240, 0.0014631, -0.0010702, 0.0010668
2: 0.0142819, 0.0163219, 0.0141489, 0.0163760, -0.0015868, 0.0015984
3: 0.0001125, 0.0016465, 0.0000125, 0.0016872, -0.0011884, 0.0011999
4: -0.0042759, -0.0028609, -0.0043681, -0.0028234, -0.0011356, 0.0011221
5: 0.0080505, 0.0095817, 0.0079506, 0.0096222, -0.0011858, 0.0011976
6: 0.0093171, 0.0098949, 0.0093017, 0.0099326, -0.0004998, 0.0005931
7: -0.0192002, -0.0158761, -0.0192883, -0.0156594, -0.0025775, 0.0025197
8: 0.9687800, 0.9783039, 0.9685276, 0.9789250, -0.0074666, 0.0074191
9: 0.0040296, 0.0068287, 0.0038471, 0.0069029, -0.0021396, 0.0021777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044759, upper bound: 0.0043340
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054820, upper bound: 0.0054609
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000816, -0.0002892, 0.0000509, -0.0002404, 0.0002843
1: 0.0000593, 0.0014467, -0.0000339, 0.0013996, -0.0010342, 0.0011292
2: 0.0141734, 0.0162511, 0.0142439, 0.0163907, -0.0016858, 0.0015362
3: 0.0000309, 0.0015933, 0.0000839, 0.0016982, -0.0012653, 0.0011497
4: -0.0043511, -0.0029100, -0.0043022, -0.0028132, -0.0011854, 0.0011082
5: 0.0079690, 0.0095285, 0.0080219, 0.0096333, -0.0012627, 0.0011471
6: 0.0093371, 0.0099256, 0.0092976, 0.0099057, -0.0005685, 0.0005340
7: -0.0190849, -0.0156993, -0.0193123, -0.0158142, -0.0024344, 0.0027142
8: 0.9691105, 0.9788105, 0.9684589, 0.9784814, -0.0071848, 0.0078754
9: 0.0038807, 0.0067316, 0.0039775, 0.0069231, -0.0022945, 0.0020678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040984, upper bound: 0.0046410
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0056042
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000794, -0.0002892, 0.0000509, -0.0002461, 0.0002831
1: 0.0000401, 0.0014433, -0.0000339, 0.0013996, -0.0010614, 0.0011338
2: 0.0141785, 0.0162800, 0.0142439, 0.0163907, -0.0016927, 0.0015768
3: 0.0000347, 0.0016150, 0.0000839, 0.0016982, -0.0012705, 0.0011801
4: -0.0043476, -0.0028900, -0.0043022, -0.0028132, -0.0011902, 0.0011334
5: 0.0079728, 0.0095502, 0.0080219, 0.0096333, -0.0012680, 0.0011774
6: 0.0093289, 0.0099242, 0.0092976, 0.0099057, -0.0005767, 0.0005360
7: -0.0191318, -0.0157076, -0.0193123, -0.0158142, -0.0024996, 0.0027256
8: 0.9689760, 0.9787869, 0.9684589, 0.9784814, -0.0073747, 0.0079079
9: 0.0038877, 0.0067711, 0.0039775, 0.0069231, -0.0023040, 0.0021224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040984, upper bound: 0.0046410
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0056248
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000816, -0.0002821, 0.0000923, -0.0002725, 0.0002744
1: 0.0000593, 0.0014467, -0.0000004, 0.0014631, -0.0010735, 0.0011060
2: 0.0141734, 0.0162511, 0.0141488, 0.0163406, -0.0016445, 0.0015939
3: 0.0000309, 0.0015933, 0.0000124, 0.0016605, -0.0012315, 0.0011924
4: -0.0043511, -0.0029100, -0.0043682, -0.0028479, -0.0011790, 0.0011534
5: 0.0079690, 0.0095285, 0.0079506, 0.0095957, -0.0012287, 0.0011896
6: 0.0093371, 0.0099256, 0.0093118, 0.0099326, -0.0005955, 0.0006139
7: -0.0190849, -0.0156993, -0.0192306, -0.0156593, -0.0025201, 0.0026123
8: 0.9691105, 0.9788105, 0.9686929, 0.9789252, -0.0074554, 0.0076893
9: 0.0038807, 0.0067316, 0.0038470, 0.0068543, -0.0022176, 0.0021415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0051830
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054640, upper bound: 0.0057090
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000794, -0.0002821, 0.0000923, -0.0002778, 0.0002733
1: 0.0000401, 0.0014433, -0.0000004, 0.0014631, -0.0010987, 0.0011084
2: 0.0141785, 0.0162800, 0.0141488, 0.0163406, -0.0016482, 0.0016311
3: 0.0000347, 0.0016150, 0.0000124, 0.0016605, -0.0012342, 0.0012202
4: -0.0043476, -0.0028900, -0.0043682, -0.0028479, -0.0011815, 0.0011772
5: 0.0079728, 0.0095502, 0.0079506, 0.0095957, -0.0012315, 0.0012175
6: 0.0093289, 0.0099242, 0.0093118, 0.0099326, -0.0006036, 0.0006124
7: -0.0191318, -0.0157076, -0.0192306, -0.0156593, -0.0025823, 0.0026183
8: 0.9689760, 0.9787869, 0.9686929, 0.9789252, -0.0076290, 0.0077066
9: 0.0038877, 0.0067711, 0.0038470, 0.0068543, -0.0022227, 0.0021939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0051830
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054640, upper bound: 0.0057518
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000816, -0.0002939, 0.0000514, -0.0002414, 0.0002947
1: 0.0000593, 0.0014467, -0.0000558, 0.0014004, -0.0010363, 0.0011856
2: 0.0141734, 0.0162511, 0.0142427, 0.0164236, -0.0017702, 0.0015393
3: 0.0000309, 0.0015933, 0.0000830, 0.0017229, -0.0013287, 0.0011521
4: -0.0043511, -0.0029100, -0.0043031, -0.0027904, -0.0012444, 0.0011103
5: 0.0079690, 0.0095285, 0.0080210, 0.0096580, -0.0013260, 0.0011494
6: 0.0093371, 0.0099256, 0.0092883, 0.0099060, -0.0005689, 0.0005576
7: -0.0190849, -0.0156993, -0.0193658, -0.0158122, -0.0024394, 0.0028506
8: 0.9691105, 0.9788105, 0.9683055, 0.9784872, -0.0071991, 0.0082699
9: 0.0038807, 0.0067316, 0.0039758, 0.0069682, -0.0024094, 0.0020720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0040533, upper bound: 0.0045974
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0055811
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000794, -0.0002939, 0.0000514, -0.0002422, 0.0002887
1: 0.0000401, 0.0014433, -0.0000558, 0.0014004, -0.0010416, 0.0011657
2: 0.0141785, 0.0162800, 0.0142427, 0.0164236, -0.0017404, 0.0015474
3: 0.0000347, 0.0016150, 0.0000830, 0.0017229, -0.0013062, 0.0011579
4: -0.0043476, -0.0028900, -0.0043031, -0.0027904, -0.0012238, 0.0011158
5: 0.0079728, 0.0095502, 0.0080210, 0.0096580, -0.0013036, 0.0011553
6: 0.0093289, 0.0099242, 0.0092883, 0.0099060, -0.0005771, 0.0005536
7: -0.0191318, -0.0157076, -0.0193658, -0.0158122, -0.0024521, 0.0028007
8: 0.9689760, 0.9787869, 0.9683055, 0.9784872, -0.0072368, 0.0081307
9: 0.0038877, 0.0067711, 0.0039758, 0.0069682, -0.0023681, 0.0020820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0040533, upper bound: 0.0045988
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0056248
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000816, -0.0002871, 0.0000923, -0.0002737, 0.0002802
1: 0.0000593, 0.0014467, -0.0000240, 0.0014631, -0.0010740, 0.0011326
2: 0.0141734, 0.0162511, 0.0141489, 0.0163760, -0.0016858, 0.0015947
3: 0.0000309, 0.0015933, 0.0000125, 0.0016872, -0.0012628, 0.0011930
4: -0.0043511, -0.0029100, -0.0043681, -0.0028234, -0.0012046, 0.0011539
5: 0.0079690, 0.0095285, 0.0079506, 0.0096222, -0.0012601, 0.0011903
6: 0.0093371, 0.0099256, 0.0093017, 0.0099326, -0.0005955, 0.0006239
7: -0.0190849, -0.0156993, -0.0192883, -0.0156594, -0.0025214, 0.0026796
8: 0.9691105, 0.9788105, 0.9685276, 0.9789250, -0.0074593, 0.0078809
9: 0.0038807, 0.0067316, 0.0038471, 0.0069029, -0.0022749, 0.0021427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0051744
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054640, upper bound: 0.0056991
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000794, -0.0002871, 0.0000923, -0.0002739, 0.0002742
1: 0.0000401, 0.0014433, -0.0000240, 0.0014631, -0.0010813, 0.0011150
2: 0.0141785, 0.0162800, 0.0141489, 0.0163760, -0.0016582, 0.0016065
3: 0.0000347, 0.0016150, 0.0000125, 0.0016872, -0.0012417, 0.0012022
4: -0.0043476, -0.0028900, -0.0043681, -0.0028234, -0.0011870, 0.0011589
5: 0.0079728, 0.0095502, 0.0079506, 0.0096222, -0.0012389, 0.0011995
6: 0.0093289, 0.0099242, 0.0093017, 0.0099326, -0.0006036, 0.0006224
7: -0.0191318, -0.0157076, -0.0192883, -0.0156594, -0.0025408, 0.0026312
8: 0.9689760, 0.9787869, 0.9685276, 0.9789250, -0.0075134, 0.0077534
9: 0.0038877, 0.0067711, 0.0038471, 0.0069029, -0.0022340, 0.0021589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0051751
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054640, upper bound: 0.0057518
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002757, 0.0000360, -0.0002356, 0.0002367
1: -0.0000339, 0.0013996, 0.0000295, 0.0013767, -0.0010299, 0.0009888
2: 0.0142439, 0.0163907, 0.0142782, 0.0162958, -0.0014762, 0.0015372
3: 0.0000839, 0.0016982, 0.0001097, 0.0016268, -0.0011079, 0.0011535
4: -0.0043022, -0.0028132, -0.0042785, -0.0028790, -0.0010379, 0.0010823
5: 0.0080219, 0.0096333, 0.0080477, 0.0095620, -0.0011056, 0.0011512
6: 0.0092976, 0.0099057, 0.0093245, 0.0098959, -0.0004919, 0.0004694
7: -0.0193123, -0.0158142, -0.0191576, -0.0158701, -0.0024721, 0.0023767
8: 0.9684589, 0.9784814, 0.9689021, 0.9783213, -0.0071816, 0.0068964
9: 0.0039775, 0.0069231, 0.0040245, 0.0067928, -0.0020088, 0.0020906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036264, upper bound: 0.0037568
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054778, upper bound: 0.0053000
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002821, 0.0000923, -0.0002757, 0.0000360, -0.0002359, 0.0002783
1: -0.0000004, 0.0014631, 0.0000295, 0.0013767, -0.0010571, 0.0010656
2: 0.0141488, 0.0163406, 0.0142782, 0.0162958, -0.0015912, 0.0015720
3: 0.0000124, 0.0016605, 0.0001097, 0.0016268, -0.0011943, 0.0011770
4: -0.0043682, -0.0028479, -0.0042785, -0.0028790, -0.0011177, 0.0011248
5: 0.0079506, 0.0095957, 0.0080477, 0.0095620, -0.0011920, 0.0011744
6: 0.0093118, 0.0099326, 0.0093245, 0.0098959, -0.0005842, 0.0005020
7: -0.0192306, -0.0156593, -0.0191576, -0.0158701, -0.0024949, 0.0025641
8: 0.9686929, 0.9789252, 0.9689021, 0.9783213, -0.0073506, 0.0074334
9: 0.0038470, 0.0068543, 0.0040245, 0.0067928, -0.0021667, 0.0021180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0041795, upper bound: 0.0046322
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054778, upper bound: 0.0055011
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002794, 0.0000343, -0.0002360, 0.0002425
1: -0.0000339, 0.0013996, 0.0000120, 0.0013742, -0.0010370, 0.0010141
2: 0.0142439, 0.0163907, 0.0142819, 0.0163219, -0.0015143, 0.0015478
3: 0.0000839, 0.0016982, 0.0001125, 0.0016465, -0.0011369, 0.0011615
4: -0.0043022, -0.0028132, -0.0042759, -0.0028609, -0.0010637, 0.0010897
5: 0.0080219, 0.0096333, 0.0080505, 0.0095817, -0.0011346, 0.0011592
6: 0.0092976, 0.0099057, 0.0093171, 0.0098949, -0.0004949, 0.0004750
7: -0.0193123, -0.0158142, -0.0192002, -0.0158761, -0.0024894, 0.0024422
8: 0.9684589, 0.9784814, 0.9687800, 0.9783039, -0.0072312, 0.0070742
9: 0.0039775, 0.0069231, 0.0040296, 0.0068287, -0.0020634, 0.0021051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039565, upper bound: 0.0032842
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054624, upper bound: 0.0052580
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002821, 0.0000923, -0.0002794, 0.0000343, -0.0002360, 0.0002841
1: -0.0000004, 0.0014631, 0.0000120, 0.0013742, -0.0010604, 0.0010909
2: 0.0141488, 0.0163406, 0.0142819, 0.0163219, -0.0016294, 0.0015770
3: 0.0000124, 0.0016605, 0.0001125, 0.0016465, -0.0012233, 0.0011807
4: -0.0043682, -0.0028479, -0.0042759, -0.0028609, -0.0011434, 0.0011282
5: 0.0079506, 0.0095957, 0.0080505, 0.0095817, -0.0012209, 0.0011781
6: 0.0093118, 0.0099326, 0.0093171, 0.0098949, -0.0005831, 0.0005076
7: -0.0192306, -0.0156593, -0.0192002, -0.0158761, -0.0025029, 0.0026296
8: 0.9686929, 0.9789252, 0.9687800, 0.9783039, -0.0073736, 0.0076112
9: 0.0038470, 0.0068543, 0.0040296, 0.0068287, -0.0022212, 0.0021248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039565, upper bound: 0.0045005
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054624, upper bound: 0.0054820
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002939, 0.0000514, -0.0002757, 0.0000360, -0.0002466, 0.0002383
1: -0.0000558, 0.0014004, 0.0000295, 0.0013767, -0.0010869, 0.0009953
2: 0.0142427, 0.0164236, 0.0142782, 0.0162958, -0.0014860, 0.0016224
3: 0.0000830, 0.0017229, 0.0001097, 0.0016268, -0.0011152, 0.0012175
4: -0.0043031, -0.0027904, -0.0042785, -0.0028790, -0.0010447, 0.0011418
5: 0.0080210, 0.0096580, 0.0080477, 0.0095620, -0.0011130, 0.0012150
6: 0.0092883, 0.0099060, 0.0093245, 0.0098959, -0.0005158, 0.0004722
7: -0.0193658, -0.0158122, -0.0191576, -0.0158701, -0.0026096, 0.0023927
8: 0.9683055, 0.9784872, 0.9689021, 0.9783213, -0.0075796, 0.0069421
9: 0.0039758, 0.0069682, 0.0040245, 0.0067928, -0.0020223, 0.0022065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053819, upper bound: 0.0052523
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054277, upper bound: 0.0052523
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002939, 0.0000514, -0.0002794, 0.0000343, -0.0002411, 0.0002379
1: -0.0000558, 0.0014004, 0.0000120, 0.0013742, -0.0010678, 0.0009927
2: 0.0142427, 0.0164236, 0.0142819, 0.0163219, -0.0014823, 0.0015938
3: 0.0000830, 0.0017229, 0.0001125, 0.0016465, -0.0011127, 0.0011960
4: -0.0043031, -0.0027904, -0.0042759, -0.0028609, -0.0010416, 0.0011221
5: 0.0080210, 0.0096580, 0.0080505, 0.0095817, -0.0011105, 0.0011936
6: 0.0092883, 0.0099060, 0.0093171, 0.0098949, -0.0005121, 0.0004669
7: -0.0193658, -0.0158122, -0.0192002, -0.0158761, -0.0025618, 0.0023884
8: 0.9683055, 0.9784872, 0.9687800, 0.9783039, -0.0074461, 0.0069248
9: 0.0039758, 0.0069682, 0.0040296, 0.0068287, -0.0020185, 0.0021669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053819, upper bound: 0.0052523
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054277, upper bound: 0.0052523
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000923, -0.0002757, 0.0000360, -0.0002404, 0.0002796
1: -0.0000240, 0.0014631, 0.0000295, 0.0013767, -0.0010768, 0.0010684
2: 0.0141489, 0.0163760, 0.0142782, 0.0162958, -0.0015955, 0.0016021
3: 0.0000125, 0.0016872, 0.0001097, 0.0016268, -0.0011976, 0.0011998
4: -0.0043681, -0.0028234, -0.0042785, -0.0028790, -0.0011206, 0.0011468
5: 0.0079506, 0.0096222, 0.0080477, 0.0095620, -0.0011952, 0.0011971
6: 0.0093017, 0.0099326, 0.0093245, 0.0098959, -0.0005942, 0.0005032
7: -0.0192883, -0.0156594, -0.0191576, -0.0158701, -0.0025458, 0.0025711
8: 0.9685276, 0.9789250, 0.9689021, 0.9783213, -0.0074902, 0.0074533
9: 0.0038471, 0.0069029, 0.0040245, 0.0067928, -0.0021725, 0.0021609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0043340, upper bound: 0.0044759
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054989
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000923, -0.0002794, 0.0000343, -0.0002368, 0.0002796
1: -0.0000240, 0.0014631, 0.0000120, 0.0013742, -0.0010668, 0.0010702
2: 0.0141489, 0.0163760, 0.0142819, 0.0163219, -0.0015984, 0.0015868
3: 0.0000125, 0.0016872, 0.0001125, 0.0016465, -0.0011999, 0.0011884
4: -0.0043681, -0.0028234, -0.0042759, -0.0028609, -0.0011221, 0.0011356
5: 0.0079506, 0.0096222, 0.0080505, 0.0095817, -0.0011976, 0.0011858
6: 0.0093017, 0.0099326, 0.0093171, 0.0098949, -0.0005931, 0.0004998
7: -0.0192883, -0.0156594, -0.0192002, -0.0158761, -0.0025197, 0.0025775
8: 0.9685276, 0.9789250, 0.9687800, 0.9783039, -0.0074191, 0.0074666
9: 0.0038471, 0.0069029, 0.0040296, 0.0068287, -0.0021777, 0.0021396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0043340, upper bound: 0.0044759
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054989
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002693, 0.0000816, -0.0002843, 0.0002404
1: -0.0000339, 0.0013996, 0.0000593, 0.0014467, -0.0011292, 0.0010342
2: 0.0142439, 0.0163907, 0.0141734, 0.0162511, -0.0015362, 0.0016858
3: 0.0000839, 0.0016982, 0.0000309, 0.0015933, -0.0011497, 0.0012653
4: -0.0043022, -0.0028132, -0.0043511, -0.0029100, -0.0011082, 0.0011854
5: 0.0080219, 0.0096333, 0.0079690, 0.0095285, -0.0011471, 0.0012627
6: 0.0092976, 0.0099057, 0.0093371, 0.0099256, -0.0005340, 0.0005685
7: -0.0193123, -0.0158142, -0.0190849, -0.0156993, -0.0027142, 0.0024344
8: 0.9684589, 0.9784814, 0.9691105, 0.9788105, -0.0078754, 0.0071848
9: 0.0039775, 0.0069231, 0.0038807, 0.0067316, -0.0020678, 0.0022945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046410, upper bound: 0.0040984
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056042, upper bound: 0.0052580
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002734, 0.0000794, -0.0002831, 0.0002461
1: -0.0000339, 0.0013996, 0.0000401, 0.0014433, -0.0011338, 0.0010614
2: 0.0142439, 0.0163907, 0.0141785, 0.0162800, -0.0015768, 0.0016927
3: 0.0000839, 0.0016982, 0.0000347, 0.0016150, -0.0011801, 0.0012705
4: -0.0043022, -0.0028132, -0.0043476, -0.0028900, -0.0011334, 0.0011902
5: 0.0080219, 0.0096333, 0.0079728, 0.0095502, -0.0011774, 0.0012680
6: 0.0092976, 0.0099057, 0.0093289, 0.0099242, -0.0005360, 0.0005767
7: -0.0193123, -0.0158142, -0.0191318, -0.0157076, -0.0027256, 0.0024996
8: 0.9684589, 0.9784814, 0.9689760, 0.9787869, -0.0079079, 0.0073747
9: 0.0039775, 0.0069231, 0.0038877, 0.0067711, -0.0021224, 0.0023040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046410, upper bound: 0.0040984
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056042, upper bound: 0.0052580
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002821, 0.0000923, -0.0002693, 0.0000816, -0.0002744, 0.0002725
1: -0.0000004, 0.0014631, 0.0000593, 0.0014467, -0.0011060, 0.0010735
2: 0.0141488, 0.0163406, 0.0141734, 0.0162511, -0.0015939, 0.0016445
3: 0.0000124, 0.0016605, 0.0000309, 0.0015933, -0.0011924, 0.0012315
4: -0.0043682, -0.0028479, -0.0043511, -0.0029100, -0.0011534, 0.0011790
5: 0.0079506, 0.0095957, 0.0079690, 0.0095285, -0.0011896, 0.0012287
6: 0.0093118, 0.0099326, 0.0093371, 0.0099256, -0.0006139, 0.0005955
7: -0.0192306, -0.0156593, -0.0190849, -0.0156993, -0.0026123, 0.0025201
8: 0.9686929, 0.9789252, 0.9691105, 0.9788105, -0.0076893, 0.0074554
9: 0.0038470, 0.0068543, 0.0038807, 0.0067316, -0.0021415, 0.0022176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051330, upper bound: 0.0051074
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056636, upper bound: 0.0055614
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002821, 0.0000923, -0.0002734, 0.0000794, -0.0002733, 0.0002778
1: -0.0000004, 0.0014631, 0.0000401, 0.0014433, -0.0011084, 0.0010987
2: 0.0141488, 0.0163406, 0.0141785, 0.0162800, -0.0016311, 0.0016482
3: 0.0000124, 0.0016605, 0.0000347, 0.0016150, -0.0012202, 0.0012342
4: -0.0043682, -0.0028479, -0.0043476, -0.0028900, -0.0011772, 0.0011815
5: 0.0079506, 0.0095957, 0.0079728, 0.0095502, -0.0012175, 0.0012315
6: 0.0093118, 0.0099326, 0.0093289, 0.0099242, -0.0006124, 0.0006036
7: -0.0192306, -0.0156593, -0.0191318, -0.0157076, -0.0026183, 0.0025823
8: 0.9686929, 0.9789252, 0.9689760, 0.9787869, -0.0077066, 0.0076290
9: 0.0038470, 0.0068543, 0.0038877, 0.0067711, -0.0021939, 0.0022227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051330, upper bound: 0.0051074
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056636, upper bound: 0.0055614
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002939, 0.0000514, -0.0002693, 0.0000816, -0.0002947, 0.0002414
1: -0.0000558, 0.0014004, 0.0000593, 0.0014467, -0.0011856, 0.0010363
2: 0.0142427, 0.0164236, 0.0141734, 0.0162511, -0.0015393, 0.0017702
3: 0.0000830, 0.0017229, 0.0000309, 0.0015933, -0.0011521, 0.0013287
4: -0.0043031, -0.0027904, -0.0043511, -0.0029100, -0.0011103, 0.0012444
5: 0.0080210, 0.0096580, 0.0079690, 0.0095285, -0.0011494, 0.0013260
6: 0.0092883, 0.0099060, 0.0093371, 0.0099256, -0.0005576, 0.0005689
7: -0.0193658, -0.0158122, -0.0190849, -0.0156993, -0.0028506, 0.0024394
8: 0.9683055, 0.9784872, 0.9691105, 0.9788105, -0.0082699, 0.0071991
9: 0.0039758, 0.0069682, 0.0038807, 0.0067316, -0.0020720, 0.0024094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0045974, upper bound: 0.0040533
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055811, upper bound: 0.0052840
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002939, 0.0000514, -0.0002734, 0.0000794, -0.0002887, 0.0002422
1: -0.0000558, 0.0014004, 0.0000401, 0.0014433, -0.0011657, 0.0010416
2: 0.0142427, 0.0164236, 0.0141785, 0.0162800, -0.0015474, 0.0017404
3: 0.0000830, 0.0017229, 0.0000347, 0.0016150, -0.0011579, 0.0013062
4: -0.0043031, -0.0027904, -0.0043476, -0.0028900, -0.0011158, 0.0012238
5: 0.0080210, 0.0096580, 0.0079728, 0.0095502, -0.0011553, 0.0013036
6: 0.0092883, 0.0099060, 0.0093289, 0.0099242, -0.0005536, 0.0005771
7: -0.0193658, -0.0158122, -0.0191318, -0.0157076, -0.0028007, 0.0024521
8: 0.9683055, 0.9784872, 0.9689760, 0.9787869, -0.0081307, 0.0072368
9: 0.0039758, 0.0069682, 0.0038877, 0.0067711, -0.0020820, 0.0023681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0045974, upper bound: 0.0040533
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055811, upper bound: 0.0052840
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000923, -0.0002693, 0.0000816, -0.0002802, 0.0002737
1: -0.0000240, 0.0014631, 0.0000593, 0.0014467, -0.0011326, 0.0010740
2: 0.0141489, 0.0163760, 0.0141734, 0.0162511, -0.0015947, 0.0016858
3: 0.0000125, 0.0016872, 0.0000309, 0.0015933, -0.0011930, 0.0012628
4: -0.0043681, -0.0028234, -0.0043511, -0.0029100, -0.0011539, 0.0012046
5: 0.0079506, 0.0096222, 0.0079690, 0.0095285, -0.0011903, 0.0012601
6: 0.0093017, 0.0099326, 0.0093371, 0.0099256, -0.0006239, 0.0005955
7: -0.0192883, -0.0156594, -0.0190849, -0.0156993, -0.0026796, 0.0025214
8: 0.9685276, 0.9789250, 0.9691105, 0.9788105, -0.0078809, 0.0074593
9: 0.0038471, 0.0069029, 0.0038807, 0.0067316, -0.0021427, 0.0022749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051149, upper bound: 0.0051122
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055927
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000923, -0.0002734, 0.0000794, -0.0002742, 0.0002739
1: -0.0000240, 0.0014631, 0.0000401, 0.0014433, -0.0011150, 0.0010813
2: 0.0141489, 0.0163760, 0.0141785, 0.0162800, -0.0016065, 0.0016582
3: 0.0000125, 0.0016872, 0.0000347, 0.0016150, -0.0012022, 0.0012417
4: -0.0043681, -0.0028234, -0.0043476, -0.0028900, -0.0011589, 0.0011870
5: 0.0079506, 0.0096222, 0.0079728, 0.0095502, -0.0011995, 0.0012389
6: 0.0093017, 0.0099326, 0.0093289, 0.0099242, -0.0006224, 0.0006036
7: -0.0192883, -0.0156594, -0.0191318, -0.0157076, -0.0026312, 0.0025408
8: 0.9685276, 0.9789250, 0.9689760, 0.9787869, -0.0077534, 0.0075134
9: 0.0038471, 0.0069029, 0.0038877, 0.0067711, -0.0021589, 0.0022340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051149, upper bound: 0.0051122
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055927
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002892, 0.0000509, -0.0002297, 0.0002297
1: -0.0000339, 0.0013996, -0.0000339, 0.0013996, -0.0009692, 0.0009692
2: 0.0142439, 0.0163907, 0.0142439, 0.0163907, -0.0014440, 0.0014440
3: 0.0000839, 0.0016982, 0.0000839, 0.0016982, -0.0010825, 0.0010825
4: -0.0043022, -0.0028132, -0.0043022, -0.0028132, -0.0010268, 0.0010268
5: 0.0080219, 0.0096333, 0.0080219, 0.0096333, -0.0010802, 0.0010802
6: 0.0092976, 0.0099057, 0.0092976, 0.0099057, -0.0004981, 0.0004981
7: -0.0193123, -0.0158142, -0.0193123, -0.0158142, -0.0023062, 0.0023062
8: 0.9684589, 0.9784814, 0.9684589, 0.9784814, -0.0067491, 0.0067491
9: 0.0039775, 0.0069231, 0.0039775, 0.0069231, -0.0019546, 0.0019546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0041795, upper bound: 0.0035899
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054778, upper bound: 0.0053000
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002821, 0.0000923, -0.0002757, 0.0002332
1: -0.0000339, 0.0013996, -0.0000004, 0.0014631, -0.0010681, 0.0010192
2: 0.0142439, 0.0163907, 0.0141488, 0.0163406, -0.0015103, 0.0015921
3: 0.0000839, 0.0016982, 0.0000124, 0.0016605, -0.0011285, 0.0011938
4: -0.0043022, -0.0028132, -0.0043682, -0.0028479, -0.0011028, 0.0011295
5: 0.0080219, 0.0096333, 0.0079506, 0.0095957, -0.0011257, 0.0011913
6: 0.0092976, 0.0099057, 0.0093118, 0.0099326, -0.0005400, 0.0005939
7: -0.0193123, -0.0158142, -0.0192306, -0.0156593, -0.0025475, 0.0023694
8: 0.9684589, 0.9784814, 0.9686929, 0.9789252, -0.0074404, 0.0070676
9: 0.0039775, 0.0069231, 0.0038470, 0.0068543, -0.0020181, 0.0021578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037005, upper bound: 0.0042395
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054778, upper bound: 0.0053000
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002935, 0.0000514, -0.0002892, 0.0000509, -0.0002355, 0.0002319
1: -0.0000536, 0.0014004, -0.0000339, 0.0013996, -0.0009952, 0.0009778
2: 0.0142427, 0.0164203, 0.0142439, 0.0163907, -0.0014568, 0.0014828
3: 0.0000830, 0.0017204, 0.0000839, 0.0016982, -0.0010921, 0.0011116
4: -0.0043031, -0.0027927, -0.0043022, -0.0028132, -0.0010356, 0.0010534
5: 0.0080210, 0.0096555, 0.0080219, 0.0096333, -0.0010898, 0.0011093
6: 0.0092892, 0.0099060, 0.0092976, 0.0099057, -0.0005065, 0.0005017
7: -0.0193604, -0.0158122, -0.0193123, -0.0158142, -0.0023721, 0.0023270
8: 0.9683210, 0.9784871, 0.9684589, 0.9784814, -0.0069304, 0.0068087
9: 0.0039758, 0.0069636, 0.0039775, 0.0069231, -0.0019721, 0.0020088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034082, upper bound: 0.0037185
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0052840
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002935, 0.0000514, -0.0002821, 0.0000923, -0.0002815, 0.0002349
1: -0.0000536, 0.0014004, -0.0000004, 0.0014631, -0.0010940, 0.0010230
2: 0.0142427, 0.0164203, 0.0141488, 0.0163406, -0.0015161, 0.0016309
3: 0.0000830, 0.0017204, 0.0000124, 0.0016605, -0.0011328, 0.0012230
4: -0.0043031, -0.0027927, -0.0043682, -0.0028479, -0.0011068, 0.0011561
5: 0.0080210, 0.0096555, 0.0079506, 0.0095957, -0.0011300, 0.0012205
6: 0.0092892, 0.0099060, 0.0093118, 0.0099326, -0.0005485, 0.0005942
7: -0.0193604, -0.0158122, -0.0192306, -0.0156593, -0.0026134, 0.0023788
8: 0.9683210, 0.9784871, 0.9686929, 0.9789252, -0.0076217, 0.0070944
9: 0.0039758, 0.0069636, 0.0038470, 0.0068543, -0.0020260, 0.0022120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034082, upper bound: 0.0040901
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0052840
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002939, 0.0000514, -0.0002319, 0.0002400
1: -0.0000339, 0.0013996, -0.0000558, 0.0014004, -0.0009778, 0.0010248
2: 0.0142439, 0.0163907, 0.0142427, 0.0164236, -0.0015265, 0.0014568
3: 0.0000839, 0.0016982, 0.0000830, 0.0017229, -0.0011441, 0.0010921
4: -0.0043022, -0.0028132, -0.0043031, -0.0027904, -0.0010863, 0.0010356
5: 0.0080219, 0.0096333, 0.0080210, 0.0096580, -0.0011416, 0.0010898
6: 0.0092976, 0.0099057, 0.0092883, 0.0099060, -0.0005017, 0.0005219
7: -0.0193123, -0.0158142, -0.0193658, -0.0158122, -0.0023270, 0.0024396
8: 0.9684589, 0.9784814, 0.9683055, 0.9784872, -0.0068087, 0.0071351
9: 0.0039775, 0.0069231, 0.0039758, 0.0069682, -0.0020667, 0.0019721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054265, upper bound: 0.0052328
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054277, upper bound: 0.0052316
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002935, 0.0000514, -0.0002939, 0.0000514, -0.0002316, 0.0002371
1: -0.0000536, 0.0014004, -0.0000558, 0.0014004, -0.0009767, 0.0010118
2: 0.0142427, 0.0164203, 0.0142427, 0.0164236, -0.0015073, 0.0014552
3: 0.0000830, 0.0017204, 0.0000830, 0.0017229, -0.0011299, 0.0010909
4: -0.0043031, -0.0027927, -0.0043031, -0.0027904, -0.0010718, 0.0010348
5: 0.0080210, 0.0096555, 0.0080210, 0.0096580, -0.0011274, 0.0010886
6: 0.0092892, 0.0099060, 0.0092883, 0.0099060, -0.0005017, 0.0005198
7: -0.0193604, -0.0158122, -0.0193658, -0.0158122, -0.0023249, 0.0024080
8: 0.9683210, 0.9784871, 0.9683055, 0.9784872, -0.0068012, 0.0070452
9: 0.0039758, 0.0069636, 0.0039758, 0.0069682, -0.0020405, 0.0019700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054265, upper bound: 0.0052563
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054277, upper bound: 0.0052526
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002871, 0.0000923, -0.0002768, 0.0002388
1: -0.0000339, 0.0013996, -0.0000240, 0.0014631, -0.0010735, 0.0010462
2: 0.0142439, 0.0163907, 0.0141489, 0.0163760, -0.0015508, 0.0016001
3: 0.0000839, 0.0016982, 0.0000125, 0.0016872, -0.0011587, 0.0011999
4: -0.0043022, -0.0028132, -0.0043681, -0.0028234, -0.0011277, 0.0011351
5: 0.0080219, 0.0096333, 0.0079506, 0.0096222, -0.0011559, 0.0011974
6: 0.0092976, 0.0099057, 0.0093017, 0.0099326, -0.0005423, 0.0006039
7: -0.0193123, -0.0158142, -0.0192883, -0.0156594, -0.0025606, 0.0024325
8: 0.9684589, 0.9784814, 0.9685276, 0.9789250, -0.0074780, 0.0072565
9: 0.0039775, 0.0069231, 0.0038471, 0.0069029, -0.0020711, 0.0021688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0045986, upper bound: 0.0040747
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055811, upper bound: 0.0052580
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002935, 0.0000514, -0.0002871, 0.0000923, -0.0002774, 0.0002354
1: -0.0000536, 0.0014004, -0.0000240, 0.0014631, -0.0010764, 0.0010324
2: 0.0142427, 0.0164203, 0.0141489, 0.0163760, -0.0015287, 0.0016044
3: 0.0000830, 0.0017204, 0.0000125, 0.0016872, -0.0011421, 0.0012031
4: -0.0043031, -0.0027927, -0.0043681, -0.0028234, -0.0011160, 0.0011384
5: 0.0080210, 0.0096555, 0.0079506, 0.0096222, -0.0011393, 0.0012006
6: 0.0092892, 0.0099060, 0.0093017, 0.0099326, -0.0005440, 0.0006043
7: -0.0193604, -0.0158122, -0.0192883, -0.0156594, -0.0025681, 0.0023965
8: 0.9683210, 0.9784871, 0.9685276, 0.9789250, -0.0074980, 0.0071544
9: 0.0039758, 0.0069636, 0.0038471, 0.0069029, -0.0020416, 0.0021748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0045986, upper bound: 0.0040747
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055811, upper bound: 0.0052840
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002821, 0.0000923, -0.0002892, 0.0000509, -0.0002332, 0.0002757
1: -0.0000004, 0.0014631, -0.0000339, 0.0013996, -0.0010192, 0.0010681
2: 0.0141488, 0.0163406, 0.0142439, 0.0163907, -0.0015921, 0.0015103
3: 0.0000124, 0.0016605, 0.0000839, 0.0016982, -0.0011938, 0.0011285
4: -0.0043682, -0.0028479, -0.0043022, -0.0028132, -0.0011295, 0.0011028
5: 0.0079506, 0.0095957, 0.0080219, 0.0096333, -0.0011913, 0.0011257
6: 0.0093118, 0.0099326, 0.0092976, 0.0099057, -0.0005939, 0.0005400
7: -0.0192306, -0.0156593, -0.0193123, -0.0158142, -0.0023694, 0.0025475
8: 0.9686929, 0.9789252, 0.9684589, 0.9784814, -0.0070676, 0.0074404
9: 0.0038470, 0.0068543, 0.0039775, 0.0069231, -0.0021578, 0.0020181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0043484, upper bound: 0.0045255
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054833
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000923, -0.0002892, 0.0000509, -0.0002388, 0.0002768
1: -0.0000240, 0.0014631, -0.0000339, 0.0013996, -0.0010462, 0.0010735
2: 0.0141489, 0.0163760, 0.0142439, 0.0163907, -0.0016001, 0.0015508
3: 0.0000125, 0.0016872, 0.0000839, 0.0016982, -0.0011999, 0.0011587
4: -0.0043681, -0.0028234, -0.0043022, -0.0028132, -0.0011351, 0.0011277
5: 0.0079506, 0.0096222, 0.0080219, 0.0096333, -0.0011974, 0.0011559
6: 0.0093017, 0.0099326, 0.0092976, 0.0099057, -0.0006039, 0.0005423
7: -0.0192883, -0.0156594, -0.0193123, -0.0158142, -0.0024325, 0.0025606
8: 0.9685276, 0.9789250, 0.9684589, 0.9784814, -0.0072565, 0.0074780
9: 0.0038471, 0.0069029, 0.0039775, 0.0069231, -0.0021688, 0.0020711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0043484, upper bound: 0.0045255
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054953
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002821, 0.0000923, -0.0002821, 0.0000923, -0.0002668, 0.0002668
1: -0.0000004, 0.0014631, -0.0000004, 0.0014631, -0.0010597, 0.0010597
2: 0.0141488, 0.0163406, 0.0141488, 0.0163406, -0.0015687, 0.0015687
3: 0.0000124, 0.0016605, 0.0000124, 0.0016605, -0.0011718, 0.0011718
4: -0.0043682, -0.0028479, -0.0043682, -0.0028479, -0.0011508, 0.0011508
5: 0.0079506, 0.0095957, 0.0079506, 0.0095957, -0.0011690, 0.0011690
6: 0.0093118, 0.0099326, 0.0093118, 0.0099326, -0.0006208, 0.0006208
7: -0.0192306, -0.0156593, -0.0192306, -0.0156593, -0.0024582, 0.0024582
8: 0.9686929, 0.9789252, 0.9686929, 0.9789252, -0.0073419, 0.0073419
9: 0.0038470, 0.0068543, 0.0038470, 0.0068543, -0.0020952, 0.0020952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052600, upper bound: 0.0050155
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055646
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000923, -0.0002821, 0.0000923, -0.0002721, 0.0002681
1: -0.0000240, 0.0014631, -0.0000004, 0.0014631, -0.0010872, 0.0010625
2: 0.0141489, 0.0163760, 0.0141488, 0.0163406, -0.0015728, 0.0016103
3: 0.0000125, 0.0016872, 0.0000124, 0.0016605, -0.0011749, 0.0012028
4: -0.0043681, -0.0028234, -0.0043682, -0.0028479, -0.0011537, 0.0011751
5: 0.0079506, 0.0096222, 0.0079506, 0.0095957, -0.0011721, 0.0011998
6: 0.0093017, 0.0099326, 0.0093118, 0.0099326, -0.0006308, 0.0006208
7: -0.0192883, -0.0156594, -0.0192306, -0.0156593, -0.0025202, 0.0024649
8: 0.9685276, 0.9789250, 0.9686929, 0.9789252, -0.0075360, 0.0073611
9: 0.0038471, 0.0069029, 0.0038470, 0.0068543, -0.0021009, 0.0021474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051249, upper bound: 0.0051270
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055920
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002821, 0.0000923, -0.0002939, 0.0000514, -0.0002349, 0.0002856
1: -0.0000004, 0.0014631, -0.0000558, 0.0014004, -0.0010230, 0.0011236
2: 0.0141488, 0.0163406, 0.0142427, 0.0164236, -0.0016744, 0.0015161
3: 0.0000124, 0.0016605, 0.0000830, 0.0017229, -0.0012553, 0.0011328
4: -0.0043682, -0.0028479, -0.0043031, -0.0027904, -0.0011889, 0.0011068
5: 0.0079506, 0.0095957, 0.0080210, 0.0096580, -0.0012527, 0.0011300
6: 0.0093118, 0.0099326, 0.0092883, 0.0099060, -0.0005942, 0.0005638
7: -0.0192306, -0.0156593, -0.0193658, -0.0158122, -0.0023788, 0.0026806
8: 0.9686929, 0.9789252, 0.9683055, 0.9784872, -0.0070944, 0.0078257
9: 0.0038470, 0.0068543, 0.0039758, 0.0069682, -0.0022697, 0.0020260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0043340, upper bound: 0.0044919
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054799
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000923, -0.0002939, 0.0000514, -0.0002354, 0.0002824
1: -0.0000240, 0.0014631, -0.0000558, 0.0014004, -0.0010324, 0.0011104
2: 0.0141489, 0.0163760, 0.0142427, 0.0164236, -0.0016550, 0.0015287
3: 0.0000125, 0.0016872, 0.0000830, 0.0017229, -0.0012409, 0.0011421
4: -0.0043681, -0.0028234, -0.0043031, -0.0027904, -0.0011742, 0.0011160
5: 0.0079506, 0.0096222, 0.0080210, 0.0096580, -0.0012383, 0.0011393
6: 0.0093017, 0.0099326, 0.0092883, 0.0099060, -0.0006043, 0.0005617
7: -0.0192883, -0.0156594, -0.0193658, -0.0158122, -0.0023965, 0.0026487
8: 0.9685276, 0.9789250, 0.9683055, 0.9784872, -0.0071544, 0.0077347
9: 0.0038471, 0.0069029, 0.0039758, 0.0069682, -0.0022432, 0.0020416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0043340, upper bound: 0.0044919
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054953
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002821, 0.0000923, -0.0002871, 0.0000923, -0.0002681, 0.0002721
1: -0.0000004, 0.0014631, -0.0000240, 0.0014631, -0.0010625, 0.0010872
2: 0.0141488, 0.0163406, 0.0141489, 0.0163760, -0.0016103, 0.0015728
3: 0.0000124, 0.0016605, 0.0000125, 0.0016872, -0.0012028, 0.0011749
4: -0.0043682, -0.0028479, -0.0043681, -0.0028234, -0.0011751, 0.0011537
5: 0.0079506, 0.0095957, 0.0079506, 0.0096222, -0.0011998, 0.0011721
6: 0.0093118, 0.0099326, 0.0093017, 0.0099326, -0.0006208, 0.0006308
7: -0.0192306, -0.0156593, -0.0192883, -0.0156594, -0.0024649, 0.0025202
8: 0.9686929, 0.9789252, 0.9685276, 0.9789250, -0.0073611, 0.0075360
9: 0.0038470, 0.0068543, 0.0038471, 0.0069029, -0.0021474, 0.0021009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052600, upper bound: 0.0050043
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055605
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000923, -0.0002871, 0.0000923, -0.0002689, 0.0002689
1: -0.0000240, 0.0014631, -0.0000240, 0.0014631, -0.0010717, 0.0010717
2: 0.0141489, 0.0163760, 0.0141489, 0.0163760, -0.0015865, 0.0015865
3: 0.0000125, 0.0016872, 0.0000125, 0.0016872, -0.0011852, 0.0011852
4: -0.0043681, -0.0028234, -0.0043681, -0.0028234, -0.0011638, 0.0011638
5: 0.0079506, 0.0096222, 0.0079506, 0.0096222, -0.0011824, 0.0011824
6: 0.0093017, 0.0099326, 0.0093017, 0.0099326, -0.0006308, 0.0006308
7: -0.0192883, -0.0156594, -0.0192883, -0.0156594, -0.0024887, 0.0024887
8: 0.9685276, 0.9789250, 0.9685276, 0.9789250, -0.0074248, 0.0074248
9: 0.0038471, 0.0069029, 0.0038471, 0.0069029, -0.0021200, 0.0021200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052600, upper bound: 0.0050060
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055920
time: 0.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.92 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0037662, upper bound: 0.0034613
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0053035, upper bound: 0.0053035
IS_A1_B1_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0034613, upper bound: 0.0042326
IS_A1_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0053035, upper bound: 0.0053035
IS_A1_B1_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0032678, upper bound: 0.0035494
IS_A1_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0052858
IS_A1_B1_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0032678, upper bound: 0.0040848
IS_A1_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0052858
IS_A1_B1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052348, upper bound: 0.0052341
IS_A1_B1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052348, upper bound: 0.0052348
IS_A1_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052348, upper bound: 0.0052572
IS_A1_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052348, upper bound: 0.0052553
IS_A1_B1_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0044759, upper bound: 0.0040537
IS_A1_B1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054820, upper bound: 0.0052597
IS_A1_B1_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0044759, upper bound: 0.0040537
IS_A1_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054820, upper bound: 0.0052858
IS_A1_B1_A2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0040984, upper bound: 0.0045137
IS_A1_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0054864
IS_A1_B1_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0040984, upper bound: 0.0045137
IS_A1_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0054989
IS_A1_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0050030
IS_A1_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054654, upper bound: 0.0055654
IS_A1_B1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0049112, upper bound: 0.0051199
IS_A1_B1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054654, upper bound: 0.0055927
IS_A1_B1_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0040537, upper bound: 0.0044759
IS_A1_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0054820
IS_A1_B1_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0040537, upper bound: 0.0044759
IS_A1_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052597, upper bound: 0.0054989
IS_A1_B1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0049914
IS_A1_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054654, upper bound: 0.0055614
IS_A1_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0049917
IS_A1_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054654, upper bound: 0.0055927
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0037568, upper bound: 0.0036264
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0053000, upper bound: 0.0054778
IS_A1_B2_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0034750, upper bound: 0.0045079
IS_A1_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0053000, upper bound: 0.0054778
IS_A1_B2_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0032842, upper bound: 0.0039565
IS_A1_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0054624
IS_A1_B2_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0032842, upper bound: 0.0043621
IS_A1_B2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0054624
IS_A1_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0053819
IS_A1_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0054277
IS_A1_B2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0054030
IS_A1_B2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052316, upper bound: 0.0054561
IS_A1_B2_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0044759, upper bound: 0.0043340
IS_A1_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054820, upper bound: 0.0054362
IS_A1_B2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0044759, upper bound: 0.0043340
IS_A1_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054820, upper bound: 0.0054609
IS_A1_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0040984, upper bound: 0.0046410
IS_A1_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0056042
IS_A1_B2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0040984, upper bound: 0.0046410
IS_A1_B2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0056248
IS_A1_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0051830
IS_A1_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054640, upper bound: 0.0057090
IS_A1_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0051830
IS_A1_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054640, upper bound: 0.0057518
IS_A1_B2_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0040533, upper bound: 0.0045974
IS_A1_B2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0055811
IS_A1_B2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0040533, upper bound: 0.0045988
IS_A1_B2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052580, upper bound: 0.0056248
IS_A1_B2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0051744
IS_A1_B2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054640, upper bound: 0.0056991
IS_A1_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0051751
IS_A1_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054640, upper bound: 0.0057518
IS_A2_B1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0036264, upper bound: 0.0037568
IS_A2_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054778, upper bound: 0.0053000
IS_A2_B1_B1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0041795, upper bound: 0.0046322
IS_A2_B1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054778, upper bound: 0.0055011
IS_A2_B1_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0039565, upper bound: 0.0032842
IS_A2_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054624, upper bound: 0.0052580
IS_A2_B1_B1_A1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0039565, upper bound: 0.0045005
IS_A2_B1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054624, upper bound: 0.0054820
IS_A2_B1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0053819, upper bound: 0.0052523
IS_A2_B1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054277, upper bound: 0.0052523
IS_A2_B1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0053819, upper bound: 0.0052523
IS_A2_B1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054277, upper bound: 0.0052523
IS_A2_B1_B1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0043340, upper bound: 0.0044759
IS_A2_B1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054989
IS_A2_B1_B1_A2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0043340, upper bound: 0.0044759
IS_A2_B1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054989
IS_A2_B1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0046410, upper bound: 0.0040984
IS_A2_B1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0056042, upper bound: 0.0052580
IS_A2_B1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0046410, upper bound: 0.0040984
IS_A2_B1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0056042, upper bound: 0.0052580
IS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0051330, upper bound: 0.0051074
IS_A2_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0056636, upper bound: 0.0055614
IS_A2_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0051330, upper bound: 0.0051074
IS_A2_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0056636, upper bound: 0.0055614
IS_A2_B1_B2_A2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0045974, upper bound: 0.0040533
IS_A2_B1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0055811, upper bound: 0.0052840
IS_A2_B1_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0045974, upper bound: 0.0040533
IS_A2_B1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0055811, upper bound: 0.0052840
IS_A2_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0051149, upper bound: 0.0051122
IS_A2_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055927
IS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0051149, upper bound: 0.0051122
IS_A2_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055927
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0041795, upper bound: 0.0035899
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054778, upper bound: 0.0053000
IS_A2_B2_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0037005, upper bound: 0.0042395
IS_A2_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054778, upper bound: 0.0053000
IS_A2_B2_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0034082, upper bound: 0.0037185
IS_A2_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0052840
IS_A2_B2_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0034082, upper bound: 0.0040901
IS_A2_B2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0052840
IS_A2_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054265, upper bound: 0.0052328
IS_A2_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054277, upper bound: 0.0052316
IS_A2_B2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054265, upper bound: 0.0052563
IS_A2_B2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054277, upper bound: 0.0052526
IS_A2_B2_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0045986, upper bound: 0.0040747
IS_A2_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0055811, upper bound: 0.0052580
IS_A2_B2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0045986, upper bound: 0.0040747
IS_A2_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0055811, upper bound: 0.0052840
IS_A2_B2_A2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0043484, upper bound: 0.0045255
IS_A2_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054833
IS_A2_B2_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0043484, upper bound: 0.0045255
IS_A2_B2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054953
IS_A2_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052600, upper bound: 0.0050155
IS_A2_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055646
IS_A2_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0051249, upper bound: 0.0051270
IS_A2_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055920
IS_A2_B2_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0043340, upper bound: 0.0044919
IS_A2_B2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054799
IS_A2_B2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0043340, upper bound: 0.0044919
IS_A2_B2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0054362, upper bound: 0.0054953
IS_A2_B2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052600, upper bound: 0.0050043
IS_A2_B2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055605
IS_A2_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0052600, upper bound: 0.0050060
IS_A2_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.92
Output dim: 8, lower bound: -0.0056477, upper bound: 0.0055920

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000278, -0.0002757, 0.0000360, -0.0002093, 0.0001964
1: 0.0000297, 0.0013642, 0.0000295, 0.0013767, -0.0008994, 0.0008670
2: 0.0142969, 0.0162955, 0.0142782, 0.0162958, -0.0012939, 0.0013435
3: 0.0001238, 0.0016266, 0.0001097, 0.0016268, -0.0009708, 0.0010088
4: -0.0042655, -0.0028792, -0.0042785, -0.0028790, -0.0009123, 0.0009422
5: 0.0080617, 0.0095618, 0.0080477, 0.0095620, -0.0009689, 0.0010068
6: 0.0093245, 0.0098906, 0.0093245, 0.0098959, -0.0004150, 0.0004204
7: -0.0191571, -0.0159006, -0.0191576, -0.0158701, -0.0021686, 0.0020803
8: 0.9689034, 0.9782339, 0.9689021, 0.9783213, -0.0062758, 0.0060453
9: 0.0040502, 0.0067924, 0.0040245, 0.0067928, -0.0017590, 0.0018316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034613, upper bound: 0.0037662
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0034613, upper bound: 0.0053035
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002693, 0.0000731, -0.0002469, 0.0002128
1: 0.0000295, 0.0013767, 0.0000596, 0.0014336, -0.0009739, 0.0009447
2: 0.0142782, 0.0162958, 0.0141930, 0.0162507, -0.0014042, 0.0014539
3: 0.0001097, 0.0016268, 0.0000456, 0.0015930, -0.0010513, 0.0010912
4: -0.0042785, -0.0028790, -0.0043375, -0.0029103, -0.0010095, 0.0010234
5: 0.0080477, 0.0095620, 0.0079837, 0.0095282, -0.0010489, 0.0010890
6: 0.0093245, 0.0098959, 0.0093372, 0.0099201, -0.0004657, 0.0005318
7: -0.0191576, -0.0158701, -0.0190842, -0.0157313, -0.0023411, 0.0022292
8: 0.9689021, 0.9783213, 0.9691125, 0.9787189, -0.0067926, 0.0065662
9: 0.0040245, 0.0067928, 0.0039077, 0.0067310, -0.0018927, 0.0019787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0042891, upper bound: 0.0034985
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042891, upper bound: 0.0053035
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002757, 0.0000278, -0.0002014, 0.0002097
1: 0.0000120, 0.0013742, 0.0000297, 0.0013642, -0.0008905, 0.0009065
2: 0.0142819, 0.0163219, 0.0142969, 0.0162955, -0.0013542, 0.0013292
3: 0.0001125, 0.0016465, 0.0001238, 0.0016266, -0.0010168, 0.0009974
4: -0.0042759, -0.0028609, -0.0042655, -0.0028792, -0.0009496, 0.0009359
5: 0.0080505, 0.0095817, 0.0080617, 0.0095618, -0.0010148, 0.0009954
6: 0.0093171, 0.0098949, 0.0093245, 0.0098906, -0.0004226, 0.0004180
7: -0.0192002, -0.0158761, -0.0191571, -0.0159006, -0.0021372, 0.0021860
8: 0.9687800, 0.9783039, 0.9689034, 0.9782339, -0.0062098, 0.0063254
9: 0.0040296, 0.0068287, 0.0040502, 0.0067924, -0.0018462, 0.0018074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051686, upper bound: 0.0051948
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051686, upper bound: 0.0051941
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002693, 0.0000731, -0.0002518, 0.0002129
1: 0.0000120, 0.0013742, 0.0000596, 0.0014336, -0.0009974, 0.0009480
2: 0.0142819, 0.0163219, 0.0141930, 0.0162507, -0.0014091, 0.0014892
3: 0.0001125, 0.0016465, 0.0000456, 0.0015930, -0.0010550, 0.0011178
4: -0.0042759, -0.0028609, -0.0043375, -0.0029103, -0.0010129, 0.0010469
5: 0.0080505, 0.0095817, 0.0079837, 0.0095282, -0.0010526, 0.0011156
6: 0.0093171, 0.0098949, 0.0093372, 0.0099201, -0.0004679, 0.0005332
7: -0.0192002, -0.0158761, -0.0190842, -0.0157313, -0.0023980, 0.0022373
8: 0.9687800, 0.9783039, 0.9691125, 0.9787189, -0.0069571, 0.0065892
9: 0.0040296, 0.0068287, 0.0039077, 0.0067310, -0.0018995, 0.0020270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053878, upper bound: 0.0051941
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053985, upper bound: 0.0051941
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002754, 0.0000355, -0.0002759, 0.0000285, -0.0002041, 0.0002073
1: 0.0000310, 0.0013760, 0.0000284, 0.0013653, -0.0009001, 0.0008873
2: 0.0142793, 0.0162936, 0.0142953, 0.0162974, -0.0013256, 0.0013434
3: 0.0001105, 0.0016252, 0.0001226, 0.0016280, -0.0009953, 0.0010080
4: -0.0042777, -0.0028806, -0.0042665, -0.0028779, -0.0009296, 0.0009456
5: 0.0080485, 0.0095604, 0.0080605, 0.0095632, -0.0009934, 0.0010060
6: 0.0093251, 0.0098956, 0.0093240, 0.0098911, -0.0004307, 0.0004112
7: -0.0191540, -0.0158719, -0.0191602, -0.0158980, -0.0021608, 0.0021404
8: 0.9689125, 0.9783161, 0.9688947, 0.9782411, -0.0062763, 0.0061919
9: 0.0040261, 0.0067898, 0.0040481, 0.0067950, -0.0018076, 0.0018269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036869, upper bound: 0.0039644
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029618, upper bound: 0.0030679
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002753, 0.0000356, -0.0002756, 0.0000312, -0.0002068, 0.0002078
1: 0.0000315, 0.0013761, 0.0000302, 0.0013694, -0.0009039, 0.0008909
2: 0.0142791, 0.0162929, 0.0142891, 0.0162948, -0.0013306, 0.0013491
3: 0.0001104, 0.0016247, 0.0001179, 0.0016261, -0.0009989, 0.0010124
4: -0.0042778, -0.0028810, -0.0042708, -0.0028797, -0.0009344, 0.0009496
5: 0.0080483, 0.0095599, 0.0080559, 0.0095613, -0.0009970, 0.0010103
6: 0.0093253, 0.0098957, 0.0093247, 0.0098928, -0.0004327, 0.0004183
7: -0.0191529, -0.0158715, -0.0191560, -0.0158879, -0.0021705, 0.0021462
8: 0.9689157, 0.9783171, 0.9689068, 0.9782701, -0.0063033, 0.0062159
9: 0.0040257, 0.0067888, 0.0040396, 0.0067915, -0.0018131, 0.0018348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036858, upper bound: 0.0039426
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029071, upper bound: 0.0028769
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002791, 0.0000338, -0.0002759, 0.0000285, -0.0002037, 0.0002021
1: 0.0000136, 0.0013735, 0.0000284, 0.0013653, -0.0008969, 0.0008690
2: 0.0142830, 0.0163197, 0.0142953, 0.0162974, -0.0012981, 0.0013388
3: 0.0001133, 0.0016448, 0.0001226, 0.0016280, -0.0009745, 0.0010048
4: -0.0042751, -0.0028625, -0.0042665, -0.0028779, -0.0009108, 0.0009418
5: 0.0080513, 0.0095800, 0.0080605, 0.0095632, -0.0009726, 0.0010028
6: 0.0093177, 0.0098946, 0.0093240, 0.0098911, -0.0004252, 0.0004031
7: -0.0191965, -0.0158780, -0.0191602, -0.0158980, -0.0021548, 0.0020935
8: 0.9687906, 0.9782987, 0.9688947, 0.9782411, -0.0062548, 0.0060636
9: 0.0040312, 0.0068256, 0.0040481, 0.0067950, -0.0017687, 0.0018217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036002, upper bound: 0.0038845
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0028311, upper bound: 0.0029504
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000339, -0.0002756, 0.0000312, -0.0002062, 0.0002026
1: 0.0000140, 0.0013736, 0.0000302, 0.0013694, -0.0008996, 0.0008726
2: 0.0142828, 0.0163191, 0.0142891, 0.0162948, -0.0013030, 0.0013430
3: 0.0001132, 0.0016444, 0.0001179, 0.0016261, -0.0009780, 0.0010079
4: -0.0042752, -0.0028629, -0.0042708, -0.0028797, -0.0009156, 0.0009448
5: 0.0080511, 0.0095795, 0.0080559, 0.0095613, -0.0009761, 0.0010059
6: 0.0093179, 0.0098946, 0.0093247, 0.0098928, -0.0004267, 0.0004102
7: -0.0191956, -0.0158776, -0.0191560, -0.0158879, -0.0021619, 0.0020986
8: 0.9687933, 0.9782997, 0.9689068, 0.9782701, -0.0062740, 0.0060871
9: 0.0040309, 0.0068248, 0.0040396, 0.0067915, -0.0017737, 0.0018277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036002, upper bound: 0.0038765
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0027954, upper bound: 0.0027954
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002734, 0.0000713, -0.0002458, 0.0002185
1: 0.0000295, 0.0013767, 0.0000404, 0.0014309, -0.0009762, 0.0009720
2: 0.0142782, 0.0162958, 0.0141971, 0.0162795, -0.0014447, 0.0014574
3: 0.0001097, 0.0016268, 0.0000487, 0.0016146, -0.0010816, 0.0010938
4: -0.0042785, -0.0028790, -0.0043347, -0.0028903, -0.0010363, 0.0010258
5: 0.0080477, 0.0095620, 0.0079868, 0.0095498, -0.0010792, 0.0010916
6: 0.0093245, 0.0098959, 0.0093291, 0.0099189, -0.0004667, 0.0005330
7: -0.0191576, -0.0158701, -0.0191311, -0.0157379, -0.0023468, 0.0022946
8: 0.9689021, 0.9783213, 0.9689779, 0.9786999, -0.0068089, 0.0067556
9: 0.0040245, 0.0067928, 0.0039133, 0.0067705, -0.0019476, 0.0019835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0041421, upper bound: 0.0033299
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0041421, upper bound: 0.0033299
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002734, 0.0000713, -0.0002475, 0.0002136
1: 0.0000120, 0.0013742, 0.0000404, 0.0014309, -0.0009820, 0.0009540
2: 0.0142819, 0.0163219, 0.0141971, 0.0162795, -0.0014179, 0.0014662
3: 0.0001125, 0.0016465, 0.0000487, 0.0016146, -0.0010615, 0.0011005
4: -0.0042759, -0.0028609, -0.0043347, -0.0028903, -0.0010199, 0.0010310
5: 0.0080505, 0.0095817, 0.0079868, 0.0095498, -0.0010591, 0.0010983
6: 0.0093171, 0.0098949, 0.0093291, 0.0099189, -0.0004663, 0.0005298
7: -0.0192002, -0.0158761, -0.0191311, -0.0157379, -0.0023628, 0.0022513
8: 0.9687800, 0.9783039, 0.9689779, 0.9786999, -0.0068495, 0.0066307
9: 0.0040296, 0.0068287, 0.0039133, 0.0067705, -0.0019108, 0.0019967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053874, upper bound: 0.0051940
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053985, upper bound: 0.0051940
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000731, -0.0002757, 0.0000360, -0.0002128, 0.0002469
1: 0.0000596, 0.0014336, 0.0000295, 0.0013767, -0.0009447, 0.0009739
2: 0.0141930, 0.0162507, 0.0142782, 0.0162958, -0.0014539, 0.0014042
3: 0.0000456, 0.0015930, 0.0001097, 0.0016268, -0.0010912, 0.0010513
4: -0.0043375, -0.0029103, -0.0042785, -0.0028790, -0.0010234, 0.0010095
5: 0.0079837, 0.0095282, 0.0080477, 0.0095620, -0.0010890, 0.0010489
6: 0.0093372, 0.0099201, 0.0093245, 0.0098959, -0.0005318, 0.0004657
7: -0.0190842, -0.0157313, -0.0191576, -0.0158701, -0.0022292, 0.0023411
8: 0.9691125, 0.9787189, 0.9689021, 0.9783213, -0.0065662, 0.0067926
9: 0.0039077, 0.0067310, 0.0040245, 0.0067928, -0.0019787, 0.0018927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034985, upper bound: 0.0042891
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0034985, upper bound: 0.0055011
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000713, -0.0002757, 0.0000360, -0.0002185, 0.0002458
1: 0.0000404, 0.0014309, 0.0000295, 0.0013767, -0.0009720, 0.0009762
2: 0.0141971, 0.0162795, 0.0142782, 0.0162958, -0.0014574, 0.0014447
3: 0.0000487, 0.0016146, 0.0001097, 0.0016268, -0.0010938, 0.0010816
4: -0.0043347, -0.0028903, -0.0042785, -0.0028790, -0.0010258, 0.0010363
5: 0.0079868, 0.0095498, 0.0080477, 0.0095620, -0.0010916, 0.0010792
6: 0.0093291, 0.0099189, 0.0093245, 0.0098959, -0.0005330, 0.0004667
7: -0.0191311, -0.0157379, -0.0191576, -0.0158701, -0.0022946, 0.0023468
8: 0.9689779, 0.9786999, 0.9689021, 0.9783213, -0.0067556, 0.0068089
9: 0.0039133, 0.0067705, 0.0040245, 0.0067928, -0.0019835, 0.0019476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033299, upper bound: 0.0041421
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0033299, upper bound: 0.0054989
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002866, 0.0000311, -0.0002693, 0.0000754, -0.0002549, 0.0001998
1: -0.0000214, 0.0013692, 0.0000596, 0.0014371, -0.0010040, 0.0008978
2: 0.0142894, 0.0163721, 0.0141877, 0.0162507, -0.0013320, 0.0014997
3: 0.0001181, 0.0016842, 0.0000417, 0.0015929, -0.0009959, 0.0011260
4: -0.0042707, -0.0028261, -0.0043412, -0.0029103, -0.0009660, 0.0010527
5: 0.0080561, 0.0096193, 0.0079798, 0.0095282, -0.0009935, 0.0011238
6: 0.0093029, 0.0098928, 0.0093372, 0.0099216, -0.0004704, 0.0005280
7: -0.0192819, -0.0158884, -0.0190841, -0.0157227, -0.0024202, 0.0021004
8: 0.9685459, 0.9782689, 0.9691127, 0.9787437, -0.0070057, 0.0062314
9: 0.0040399, 0.0068975, 0.0039004, 0.0067309, -0.0017865, 0.0020444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050196, upper bound: 0.0049619
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049892, upper bound: 0.0049584
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000731, -0.0002693, 0.0000816, -0.0002487, 0.0002385
1: 0.0000596, 0.0014336, 0.0000593, 0.0014467, -0.0009842, 0.0009685
2: 0.0141930, 0.0162507, 0.0141734, 0.0162511, -0.0014357, 0.0014622
3: 0.0000456, 0.0015930, 0.0000309, 0.0015933, -0.0010733, 0.0010942
4: -0.0043375, -0.0029103, -0.0043511, -0.0029100, -0.0010476, 0.0010541
5: 0.0079837, 0.0095282, 0.0079690, 0.0095285, -0.0010707, 0.0010917
6: 0.0093372, 0.0099201, 0.0093371, 0.0099256, -0.0005588, 0.0005755
7: -0.0190842, -0.0157313, -0.0190849, -0.0156993, -0.0023168, 0.0022629
8: 0.9691125, 0.9787189, 0.9691105, 0.9788105, -0.0068382, 0.0067173
9: 0.0039077, 0.0067310, 0.0038807, 0.0067316, -0.0019236, 0.0019677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049972, upper bound: 0.0051724
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049972, upper bound: 0.0051724
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000728, -0.0002866, 0.0000311, -0.0002052, 0.0002535
1: 0.0000404, 0.0014332, -0.0000214, 0.0013692, -0.0009221, 0.0010055
2: 0.0141937, 0.0162795, 0.0142894, 0.0163721, -0.0015020, 0.0013684
3: 0.0000461, 0.0016146, 0.0001181, 0.0016842, -0.0011277, 0.0010236
4: -0.0043371, -0.0028903, -0.0042707, -0.0028261, -0.0010542, 0.0009903
5: 0.0079842, 0.0095498, 0.0080561, 0.0096193, -0.0011254, 0.0010213
6: 0.0093291, 0.0099199, 0.0093029, 0.0098928, -0.0005266, 0.0004711
7: -0.0191310, -0.0157323, -0.0192819, -0.0158884, -0.0021624, 0.0024238
8: 0.9689783, 0.9787159, 0.9685459, 0.9782689, -0.0064012, 0.0070160
9: 0.0039085, 0.0067705, 0.0040399, 0.0068975, -0.0020474, 0.0018384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047736, upper bound: 0.0050024
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047655, upper bound: 0.0049746
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000794, -0.0002693, 0.0000731, -0.0002430, 0.0002476
1: 0.0000401, 0.0014433, 0.0000596, 0.0014336, -0.0009879, 0.0009867
2: 0.0141785, 0.0162800, 0.0141930, 0.0162507, -0.0014659, 0.0014664
3: 0.0000347, 0.0016150, 0.0000456, 0.0015930, -0.0010970, 0.0010969
4: -0.0043476, -0.0028900, -0.0043375, -0.0029103, -0.0010566, 0.0010600
5: 0.0079728, 0.0095502, 0.0079837, 0.0095282, -0.0010945, 0.0010944
6: 0.0093289, 0.0099242, 0.0093372, 0.0099201, -0.0005722, 0.0005598
7: -0.0191318, -0.0157076, -0.0190842, -0.0157313, -0.0023158, 0.0023228
8: 0.9689760, 0.9787869, 0.9691125, 0.9787189, -0.0068595, 0.0068555
9: 0.0038877, 0.0067711, 0.0039077, 0.0067310, -0.0019727, 0.0019698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0050030
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050547, upper bound: 0.0055927
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000731, -0.0002797, 0.0000343, -0.0002129, 0.0002579
1: 0.0000596, 0.0014336, 0.0000107, 0.0013742, -0.0009480, 0.0010304
2: 0.0141930, 0.0162507, 0.0142819, 0.0163240, -0.0015384, 0.0014091
3: 0.0000456, 0.0015930, 0.0001125, 0.0016480, -0.0011547, 0.0010550
4: -0.0043375, -0.0029103, -0.0042759, -0.0028595, -0.0010816, 0.0010129
5: 0.0079837, 0.0095282, 0.0080505, 0.0095832, -0.0011524, 0.0010526
6: 0.0093372, 0.0099201, 0.0093165, 0.0098949, -0.0005332, 0.0004844
7: -0.0190842, -0.0157313, -0.0192035, -0.0158761, -0.0022373, 0.0024778
8: 0.9691125, 0.9787189, 0.9687706, 0.9783039, -0.0065892, 0.0071867
9: 0.0039077, 0.0067310, 0.0040296, 0.0068315, -0.0020943, 0.0018995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051941, upper bound: 0.0053878
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051941, upper bound: 0.0053985
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000713, -0.0002797, 0.0000343, -0.0002136, 0.0002544
1: 0.0000404, 0.0014309, 0.0000107, 0.0013742, -0.0009540, 0.0010217
2: 0.0141971, 0.0162795, 0.0142819, 0.0163240, -0.0015252, 0.0014179
3: 0.0000487, 0.0016146, 0.0001125, 0.0016480, -0.0011447, 0.0010615
4: -0.0043347, -0.0028903, -0.0042759, -0.0028595, -0.0010733, 0.0010199
5: 0.0079868, 0.0095498, 0.0080505, 0.0095832, -0.0011424, 0.0010591
6: 0.0093291, 0.0099189, 0.0093165, 0.0098949, -0.0005298, 0.0004861
7: -0.0191311, -0.0157379, -0.0192035, -0.0158761, -0.0022513, 0.0024555
8: 0.9689779, 0.9786999, 0.9687706, 0.9783039, -0.0066307, 0.0071254
9: 0.0039133, 0.0067705, 0.0040296, 0.0068315, -0.0020756, 0.0019108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051686, upper bound: 0.0054065
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051686, upper bound: 0.0054102
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002866, 0.0000311, -0.0002734, 0.0000728, -0.0002535, 0.0002052
1: -0.0000214, 0.0013692, 0.0000404, 0.0014332, -0.0010055, 0.0009221
2: 0.0142894, 0.0163721, 0.0141937, 0.0162795, -0.0013684, 0.0015020
3: 0.0001181, 0.0016842, 0.0000461, 0.0016146, -0.0010236, 0.0011277
4: -0.0042707, -0.0028261, -0.0043371, -0.0028903, -0.0009903, 0.0010542
5: 0.0080561, 0.0096193, 0.0079842, 0.0095498, -0.0010213, 0.0011254
6: 0.0093029, 0.0098928, 0.0093291, 0.0099199, -0.0004711, 0.0005266
7: -0.0192819, -0.0158884, -0.0191310, -0.0157323, -0.0024238, 0.0021624
8: 0.9685459, 0.9782689, 0.9689783, 0.9787159, -0.0070160, 0.0064012
9: 0.0040399, 0.0068975, 0.0039085, 0.0067705, -0.0018384, 0.0020474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049594, upper bound: 0.0048789
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049200, upper bound: 0.0048723
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000731, -0.0002734, 0.0000794, -0.0002476, 0.0002430
1: 0.0000596, 0.0014336, 0.0000401, 0.0014433, -0.0009867, 0.0009879
2: 0.0141930, 0.0162507, 0.0141785, 0.0162800, -0.0014664, 0.0014659
3: 0.0000456, 0.0015930, 0.0000347, 0.0016150, -0.0010969, 0.0010970
4: -0.0043375, -0.0029103, -0.0043476, -0.0028900, -0.0010600, 0.0010566
5: 0.0079837, 0.0095282, 0.0079728, 0.0095502, -0.0010944, 0.0010945
6: 0.0093372, 0.0099201, 0.0093289, 0.0099242, -0.0005598, 0.0005722
7: -0.0190842, -0.0157313, -0.0191318, -0.0157076, -0.0023228, 0.0023158
8: 0.9691125, 0.9787189, 0.9689760, 0.9787869, -0.0068555, 0.0068595
9: 0.0039077, 0.0067310, 0.0038877, 0.0067711, -0.0019698, 0.0019727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049124, upper bound: 0.0051074
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049124, upper bound: 0.0051074
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000275, -0.0002734, 0.0000728, -0.0002555, 0.0001996
1: -0.0000385, 0.0013637, 0.0000404, 0.0014332, -0.0010142, 0.0009073
2: 0.0142977, 0.0163976, 0.0141937, 0.0162795, -0.0013472, 0.0015155
3: 0.0001243, 0.0017034, 0.0000461, 0.0016146, -0.0010078, 0.0011380
4: -0.0042649, -0.0028084, -0.0043371, -0.0028903, -0.0009737, 0.0010623
5: 0.0080623, 0.0096384, 0.0079842, 0.0095498, -0.0010055, 0.0011357
6: 0.0092956, 0.0098904, 0.0093291, 0.0099199, -0.0004678, 0.0005242
7: -0.0193235, -0.0159018, -0.0191310, -0.0157323, -0.0024477, 0.0021254
8: 0.9684270, 0.9782303, 0.9689783, 0.9787159, -0.0070786, 0.0063017
9: 0.0040513, 0.0069325, 0.0039085, 0.0067705, -0.0018074, 0.0020670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049468, upper bound: 0.0048583
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049141, upper bound: 0.0048460
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000713, -0.0002734, 0.0000794, -0.0002489, 0.0002396
1: 0.0000404, 0.0014309, 0.0000401, 0.0014433, -0.0009939, 0.0009815
2: 0.0141971, 0.0162795, 0.0141785, 0.0162800, -0.0014554, 0.0014772
3: 0.0000487, 0.0016146, 0.0000347, 0.0016150, -0.0010883, 0.0011058
4: -0.0043347, -0.0028903, -0.0043476, -0.0028900, -0.0010582, 0.0010624
5: 0.0079868, 0.0095498, 0.0079728, 0.0095502, -0.0010858, 0.0011033
6: 0.0093291, 0.0099189, 0.0093289, 0.0099242, -0.0005570, 0.0005740
7: -0.0191311, -0.0157379, -0.0191318, -0.0157076, -0.0023409, 0.0022937
8: 0.9689779, 0.9786999, 0.9689760, 0.9787869, -0.0069080, 0.0068091
9: 0.0039133, 0.0067705, 0.0038877, 0.0067711, -0.0019511, 0.0019880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048995, upper bound: 0.0051122
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048995, upper bound: 0.0055927
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000278, -0.0002892, 0.0000509, -0.0002365, 0.0002235
1: 0.0000297, 0.0013642, -0.0000339, 0.0013996, -0.0009830, 0.0009971
2: 0.0142969, 0.0162955, 0.0142439, 0.0163907, -0.0014881, 0.0014688
3: 0.0001238, 0.0016266, 0.0000839, 0.0016982, -0.0011167, 0.0011029
4: -0.0042655, -0.0028792, -0.0043022, -0.0028132, -0.0010485, 0.0010291
5: 0.0080617, 0.0095618, 0.0080219, 0.0096333, -0.0011144, 0.0011008
6: 0.0093245, 0.0098906, 0.0092976, 0.0099057, -0.0004505, 0.0004787
7: -0.0191571, -0.0159006, -0.0193123, -0.0158142, -0.0023727, 0.0023916
8: 0.9689034, 0.9782339, 0.9684589, 0.9784814, -0.0068603, 0.0069525
9: 0.0040502, 0.0067924, 0.0039775, 0.0069231, -0.0020231, 0.0020034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034750, upper bound: 0.0041795
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0034750, upper bound: 0.0054778
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002820, 0.0000838, -0.0002658, 0.0002356
1: 0.0000295, 0.0013767, -0.0000001, 0.0014500, -0.0010313, 0.0010504
2: 0.0142782, 0.0162958, 0.0141685, 0.0163401, -0.0015633, 0.0015400
3: 0.0001097, 0.0016268, 0.0000272, 0.0016602, -0.0011711, 0.0011559
4: -0.0042785, -0.0028790, -0.0043545, -0.0028483, -0.0011145, 0.0010830
5: 0.0080477, 0.0095620, 0.0079653, 0.0095953, -0.0011685, 0.0011536
6: 0.0093245, 0.0098959, 0.0093119, 0.0099270, -0.0004901, 0.0005716
7: -0.0191576, -0.0158701, -0.0192298, -0.0156913, -0.0024813, 0.0024895
8: 0.9689021, 0.9783213, 0.9686952, 0.9788334, -0.0071942, 0.0073086
9: 0.0040245, 0.0067928, 0.0038740, 0.0068536, -0.0021114, 0.0020967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0042891, upper bound: 0.0036998
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0042891, upper bound: 0.0036998
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002892, 0.0000427, -0.0002287, 0.0002358
1: 0.0000120, 0.0013742, -0.0000336, 0.0013870, -0.0009736, 0.0010321
2: 0.0142819, 0.0163219, 0.0142627, 0.0163903, -0.0015415, 0.0014535
3: 0.0001125, 0.0016465, 0.0000981, 0.0016979, -0.0011573, 0.0010910
4: -0.0042759, -0.0028609, -0.0042892, -0.0028135, -0.0010820, 0.0010221
5: 0.0080505, 0.0095817, 0.0080361, 0.0096330, -0.0011550, 0.0010888
6: 0.0093171, 0.0098949, 0.0092977, 0.0099003, -0.0004578, 0.0004790
7: -0.0192002, -0.0158761, -0.0193116, -0.0158449, -0.0023398, 0.0024857
8: 0.9687800, 0.9783039, 0.9684608, 0.9783935, -0.0067905, 0.0072008
9: 0.0040296, 0.0068287, 0.0040033, 0.0069225, -0.0021003, 0.0019780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051641, upper bound: 0.0053425
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051643, upper bound: 0.0053935
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002820, 0.0000838, -0.0002708, 0.0002357
1: 0.0000120, 0.0013742, -0.0000001, 0.0014500, -0.0010549, 0.0010537
2: 0.0142819, 0.0163219, 0.0141685, 0.0163401, -0.0015682, 0.0015752
3: 0.0001125, 0.0016465, 0.0000272, 0.0016602, -0.0011748, 0.0011825
4: -0.0042759, -0.0028609, -0.0043545, -0.0028483, -0.0011179, 0.0011066
5: 0.0080505, 0.0095817, 0.0079653, 0.0095953, -0.0011722, 0.0011801
6: 0.0093171, 0.0098949, 0.0093119, 0.0099270, -0.0004923, 0.0005730
7: -0.0192002, -0.0158761, -0.0192298, -0.0156913, -0.0025382, 0.0024975
8: 0.9687800, 0.9783039, 0.9686952, 0.9788334, -0.0073587, 0.0073316
9: 0.0040296, 0.0068287, 0.0038740, 0.0068536, -0.0021182, 0.0021450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053878, upper bound: 0.0053935
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053985, upper bound: 0.0053935
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002754, 0.0000355, -0.0002895, 0.0000461, -0.0002329, 0.0002330
1: 0.0000310, 0.0013760, -0.0000350, 0.0013923, -0.0009837, 0.0010114
2: 0.0142793, 0.0162936, 0.0142549, 0.0163924, -0.0015104, 0.0014686
3: 0.0001105, 0.0016252, 0.0000922, 0.0016995, -0.0011338, 0.0011022
4: -0.0042777, -0.0028806, -0.0042946, -0.0028120, -0.0010613, 0.0010324
5: 0.0080485, 0.0095604, 0.0080302, 0.0096346, -0.0011316, 0.0011000
6: 0.0093251, 0.0098956, 0.0092971, 0.0099025, -0.0004661, 0.0004729
7: -0.0191540, -0.0158719, -0.0193150, -0.0158321, -0.0023649, 0.0024345
8: 0.9689125, 0.9783161, 0.9684511, 0.9784300, -0.0068610, 0.0070554
9: 0.0040261, 0.0067898, 0.0039926, 0.0069254, -0.0020570, 0.0019988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039350, upper bound: 0.0038514
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029618, upper bound: 0.0031552
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002753, 0.0000356, -0.0002904, 0.0000522, -0.0002393, 0.0002357
1: 0.0000315, 0.0013761, -0.0000391, 0.0014016, -0.0009993, 0.0010334
2: 0.0142791, 0.0162929, 0.0142410, 0.0163985, -0.0015430, 0.0014919
3: 0.0001104, 0.0016247, 0.0000817, 0.0017041, -0.0011581, 0.0011198
4: -0.0042778, -0.0028810, -0.0043042, -0.0028078, -0.0010860, 0.0010487
5: 0.0080483, 0.0095599, 0.0080197, 0.0096391, -0.0011558, 0.0011175
6: 0.0093253, 0.0098957, 0.0092954, 0.0099065, -0.0004731, 0.0004968
7: -0.0191529, -0.0158715, -0.0193249, -0.0158095, -0.0024032, 0.0024836
8: 0.9689157, 0.9783171, 0.9684227, 0.9784949, -0.0069700, 0.0072082
9: 0.0040257, 0.0067888, 0.0039735, 0.0069337, -0.0020996, 0.0020307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039031, upper bound: 0.0038192
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029071, upper bound: 0.0030602
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002791, 0.0000338, -0.0002895, 0.0000461, -0.0002325, 0.0002267
1: 0.0000136, 0.0013735, -0.0000350, 0.0013923, -0.0009815, 0.0009870
2: 0.0142830, 0.0163197, 0.0142549, 0.0163924, -0.0014742, 0.0014655
3: 0.0001133, 0.0016448, 0.0000922, 0.0016995, -0.0011067, 0.0011001
4: -0.0042751, -0.0028625, -0.0042946, -0.0028120, -0.0010352, 0.0010297
5: 0.0080513, 0.0095800, 0.0080302, 0.0096346, -0.0011045, 0.0010979
6: 0.0093177, 0.0098946, 0.0092971, 0.0099025, -0.0004611, 0.0004662
7: -0.0191965, -0.0158780, -0.0193150, -0.0158321, -0.0023614, 0.0023765
8: 0.9687906, 0.9782987, 0.9684511, 0.9784300, -0.0068465, 0.0068864
9: 0.0040312, 0.0068256, 0.0039926, 0.0069254, -0.0020083, 0.0019957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036002, upper bound: 0.0039858
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0028311, upper bound: 0.0030803
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002790, 0.0000339, -0.0002904, 0.0000522, -0.0002393, 0.0002310
1: 0.0000140, 0.0013736, -0.0000391, 0.0014016, -0.0009976, 0.0010178
2: 0.0142828, 0.0163191, 0.0142410, 0.0163985, -0.0015192, 0.0014898
3: 0.0001132, 0.0016444, 0.0000817, 0.0017041, -0.0011400, 0.0011183
4: -0.0042752, -0.0028629, -0.0043042, -0.0028078, -0.0010700, 0.0010466
5: 0.0080511, 0.0095795, 0.0080197, 0.0096391, -0.0011377, 0.0011161
6: 0.0093179, 0.0098946, 0.0092954, 0.0099065, -0.0004683, 0.0004932
7: -0.0191956, -0.0158776, -0.0193249, -0.0158095, -0.0024010, 0.0024437
8: 0.9687933, 0.9782997, 0.9684227, 0.9784949, -0.0069593, 0.0070976
9: 0.0040309, 0.0068248, 0.0039735, 0.0069337, -0.0020663, 0.0020291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038771, upper bound: 0.0038101
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0027955, upper bound: 0.0030122
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002757, 0.0000360, -0.0002871, 0.0000842, -0.0002666, 0.0002401
1: 0.0000295, 0.0013767, -0.0000237, 0.0014506, -0.0010344, 0.0010705
2: 0.0142782, 0.0162958, 0.0141675, 0.0163755, -0.0015938, 0.0015446
3: 0.0001097, 0.0016268, 0.0000265, 0.0016868, -0.0011943, 0.0011594
4: -0.0042785, -0.0028790, -0.0043552, -0.0028238, -0.0011362, 0.0010862
5: 0.0080477, 0.0095620, 0.0079646, 0.0096219, -0.0011917, 0.0011570
6: 0.0093245, 0.0098959, 0.0093019, 0.0099273, -0.0004914, 0.0005774
7: -0.0191576, -0.0158701, -0.0192874, -0.0156898, -0.0024888, 0.0025403
8: 0.9689021, 0.9783213, 0.9685301, 0.9788378, -0.0072157, 0.0074506
9: 0.0040245, 0.0067928, 0.0038727, 0.0069022, -0.0021544, 0.0021030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0041421, upper bound: 0.0034750
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041421, upper bound: 0.0054362
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002794, 0.0000343, -0.0002871, 0.0000842, -0.0002679, 0.0002365
1: 0.0000120, 0.0013742, -0.0000237, 0.0014506, -0.0010393, 0.0010605
2: 0.0142819, 0.0163219, 0.0141675, 0.0163755, -0.0015788, 0.0015519
3: 0.0001125, 0.0016465, 0.0000265, 0.0016868, -0.0011829, 0.0011650
4: -0.0042759, -0.0028609, -0.0043552, -0.0028238, -0.0011256, 0.0010905
5: 0.0080505, 0.0095817, 0.0079646, 0.0096219, -0.0011804, 0.0011627
6: 0.0093171, 0.0098949, 0.0093019, 0.0099273, -0.0004906, 0.0005750
7: -0.0192002, -0.0158761, -0.0192874, -0.0156898, -0.0025026, 0.0025142
8: 0.9687800, 0.9783039, 0.9685301, 0.9788378, -0.0072499, 0.0073803
9: 0.0040296, 0.0068287, 0.0038727, 0.0069022, -0.0021330, 0.0021144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053874, upper bound: 0.0053915
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053985, upper bound: 0.0053915
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002866, 0.0000311, -0.0002892, 0.0000445, -0.0002483, 0.0002356
1: -0.0000214, 0.0013692, -0.0000336, 0.0013899, -0.0010588, 0.0010381
2: 0.0142894, 0.0163721, 0.0142585, 0.0163903, -0.0015502, 0.0015819
3: 0.0001181, 0.0016842, 0.0000949, 0.0016979, -0.0011636, 0.0011879
4: -0.0042707, -0.0028261, -0.0042921, -0.0028135, -0.0010890, 0.0011096
5: 0.0080561, 0.0096193, 0.0080329, 0.0096330, -0.0011613, 0.0011855
6: 0.0093029, 0.0098928, 0.0092977, 0.0099015, -0.0004860, 0.0004849
7: -0.0192819, -0.0158884, -0.0193116, -0.0158380, -0.0025547, 0.0024979
8: 0.9685459, 0.9782689, 0.9684608, 0.9784132, -0.0073891, 0.0072417
9: 0.0040399, 0.0068975, 0.0039975, 0.0069225, -0.0021111, 0.0021573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041017, upper bound: 0.0046622
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039842, upper bound: 0.0046008
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000731, -0.0002892, 0.0000509, -0.0002401, 0.0002740
1: 0.0000596, 0.0014336, -0.0000339, 0.0013996, -0.0010263, 0.0011039
2: 0.0141930, 0.0162507, 0.0142439, 0.0163907, -0.0016482, 0.0015265
3: 0.0000456, 0.0015930, 0.0000839, 0.0016982, -0.0012370, 0.0011432
4: -0.0043375, -0.0029103, -0.0043022, -0.0028132, -0.0011595, 0.0010943
5: 0.0079837, 0.0095282, 0.0080219, 0.0096333, -0.0012345, 0.0011407
6: 0.0093372, 0.0099201, 0.0092976, 0.0099057, -0.0005665, 0.0005241
7: -0.0190842, -0.0157313, -0.0193123, -0.0158142, -0.0024285, 0.0026524
8: 0.9691125, 0.9787189, 0.9684589, 0.9784814, -0.0071371, 0.0076998
9: 0.0039077, 0.0067310, 0.0039775, 0.0069231, -0.0022427, 0.0020605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052056, upper bound: 0.0055003
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052068, upper bound: 0.0055517
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000275, -0.0002892, 0.0000445, -0.0002538, 0.0002338
1: -0.0000385, 0.0013637, -0.0000336, 0.0013899, -0.0010827, 0.0010415
2: 0.0142977, 0.0163976, 0.0142585, 0.0163903, -0.0015553, 0.0016183
3: 0.0001243, 0.0017034, 0.0000949, 0.0016979, -0.0011675, 0.0012155
4: -0.0042649, -0.0028084, -0.0042921, -0.0028135, -0.0010925, 0.0011321
5: 0.0080623, 0.0096384, 0.0080329, 0.0096330, -0.0011652, 0.0012131
6: 0.0092956, 0.0098904, 0.0092977, 0.0099015, -0.0004870, 0.0004863
7: -0.0193235, -0.0159018, -0.0193116, -0.0158380, -0.0026168, 0.0025062
8: 0.9684270, 0.9782303, 0.9684608, 0.9784132, -0.0075585, 0.0072656
9: 0.0040513, 0.0069325, 0.0039975, 0.0069225, -0.0021181, 0.0022091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039674, upper bound: 0.0045226
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038172, upper bound: 0.0044426
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000713, -0.0002892, 0.0000509, -0.0002458, 0.0002729
1: 0.0000404, 0.0014309, -0.0000339, 0.0013996, -0.0010536, 0.0011063
2: 0.0141971, 0.0162795, 0.0142439, 0.0163907, -0.0016517, 0.0015670
3: 0.0000487, 0.0016146, 0.0000839, 0.0016982, -0.0012396, 0.0011735
4: -0.0043347, -0.0028903, -0.0043022, -0.0028132, -0.0011619, 0.0011211
5: 0.0079868, 0.0095498, 0.0080219, 0.0096333, -0.0012372, 0.0011709
6: 0.0093291, 0.0099189, 0.0092976, 0.0099057, -0.0005676, 0.0005251
7: -0.0191311, -0.0157379, -0.0193123, -0.0158142, -0.0024938, 0.0026581
8: 0.9689779, 0.9786999, 0.9684589, 0.9784814, -0.0073265, 0.0077161
9: 0.0039133, 0.0067705, 0.0039775, 0.0069231, -0.0022475, 0.0021154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033387, upper bound: 0.0044515
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0033387, upper bound: 0.0056248
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002866, 0.0000311, -0.0002820, 0.0000860, -0.0002782, 0.0002251
1: -0.0000214, 0.0013692, -0.0000000, 0.0014534, -0.0010847, 0.0010116
2: 0.0142894, 0.0163721, 0.0141634, 0.0163400, -0.0015041, 0.0016206
3: 0.0001181, 0.0016842, 0.0000233, 0.0016601, -0.0011263, 0.0012168
4: -0.0042707, -0.0028261, -0.0043581, -0.0028483, -0.0010782, 0.0011365
5: 0.0080561, 0.0096193, 0.0079615, 0.0095953, -0.0011238, 0.0012145
6: 0.0093029, 0.0098928, 0.0093119, 0.0099285, -0.0005047, 0.0005662
7: -0.0192819, -0.0158884, -0.0192297, -0.0156829, -0.0026170, 0.0023889
8: 0.9685459, 0.9782689, 0.9686955, 0.9788574, -0.0075696, 0.0070329
9: 0.0040399, 0.0068975, 0.0038670, 0.0068536, -0.0020281, 0.0022101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050196, upper bound: 0.0051611
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049892, upper bound: 0.0051563
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000731, -0.0002821, 0.0000923, -0.0002721, 0.0002642
1: 0.0000596, 0.0014336, -0.0000004, 0.0014631, -0.0010648, 0.0010831
2: 0.0141930, 0.0162507, 0.0141488, 0.0163406, -0.0016107, 0.0015828
3: 0.0000456, 0.0015930, 0.0000124, 0.0016605, -0.0012064, 0.0011849
4: -0.0043375, -0.0029103, -0.0043682, -0.0028479, -0.0011544, 0.0011378
5: 0.0079837, 0.0095282, 0.0079506, 0.0095957, -0.0012038, 0.0011823
6: 0.0093372, 0.0099201, 0.0093118, 0.0099326, -0.0005930, 0.0006083
7: -0.0190842, -0.0157313, -0.0192306, -0.0156593, -0.0025134, 0.0025567
8: 0.9691125, 0.9787189, 0.9686929, 0.9789252, -0.0074016, 0.0075308
9: 0.0039077, 0.0067310, 0.0038470, 0.0068543, -0.0021718, 0.0021333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049967, upper bound: 0.0053641
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049967, upper bound: 0.0057105
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000275, -0.0002820, 0.0000860, -0.0002830, 0.0002229
1: -0.0000385, 0.0013637, -0.0000000, 0.0014534, -0.0011062, 0.0010126
2: 0.0142977, 0.0163976, 0.0141634, 0.0163400, -0.0015056, 0.0016532
3: 0.0001243, 0.0017034, 0.0000233, 0.0016601, -0.0011275, 0.0012415
4: -0.0042649, -0.0028084, -0.0043581, -0.0028483, -0.0010793, 0.0011575
5: 0.0080623, 0.0096384, 0.0079615, 0.0095953, -0.0011250, 0.0012391
6: 0.0092956, 0.0098904, 0.0093119, 0.0099285, -0.0005030, 0.0005667
7: -0.0193235, -0.0159018, -0.0192297, -0.0156829, -0.0026712, 0.0023915
8: 0.9684270, 0.9782303, 0.9686955, 0.9788574, -0.0077216, 0.0070402
9: 0.0040513, 0.0069325, 0.0038670, 0.0068536, -0.0020302, 0.0022556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049469, upper bound: 0.0050811
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049151, upper bound: 0.0050599
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000713, -0.0002821, 0.0000923, -0.0002775, 0.0002623
1: 0.0000404, 0.0014309, -0.0000004, 0.0014631, -0.0010901, 0.0010826
2: 0.0141971, 0.0162795, 0.0141488, 0.0163406, -0.0016100, 0.0016203
3: 0.0000487, 0.0016146, 0.0000124, 0.0016605, -0.0012059, 0.0012134
4: -0.0043347, -0.0028903, -0.0043682, -0.0028479, -0.0011539, 0.0011635
5: 0.0079868, 0.0095498, 0.0079506, 0.0095957, -0.0012032, 0.0012107
6: 0.0093291, 0.0099189, 0.0093118, 0.0099326, -0.0005935, 0.0006071
7: -0.0191311, -0.0157379, -0.0192306, -0.0156593, -0.0025759, 0.0025555
8: 0.9689779, 0.9786999, 0.9686929, 0.9789252, -0.0075770, 0.0075274
9: 0.0039133, 0.0067705, 0.0038470, 0.0068543, -0.0021708, 0.0021860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049112, upper bound: 0.0053249
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049112, upper bound: 0.0057518
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000731, -0.0002939, 0.0000514, -0.0002411, 0.0002838
1: 0.0000596, 0.0014336, -0.0000558, 0.0014004, -0.0010284, 0.0011577
2: 0.0141930, 0.0162507, 0.0142427, 0.0164236, -0.0017277, 0.0015296
3: 0.0000456, 0.0015930, 0.0000830, 0.0017229, -0.0012964, 0.0011456
4: -0.0043375, -0.0029103, -0.0043031, -0.0027904, -0.0012167, 0.0010965
5: 0.0079837, 0.0095282, 0.0080210, 0.0096580, -0.0012938, 0.0011430
6: 0.0093372, 0.0099201, 0.0092883, 0.0099060, -0.0005674, 0.0005491
7: -0.0190842, -0.0157313, -0.0193658, -0.0158122, -0.0024335, 0.0027768
8: 0.9691125, 0.9787189, 0.9683055, 0.9784872, -0.0071515, 0.0080722
9: 0.0039077, 0.0067310, 0.0039758, 0.0069682, -0.0023489, 0.0020647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051904, upper bound: 0.0054633
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051904, upper bound: 0.0055102
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000713, -0.0002939, 0.0000514, -0.0002418, 0.0002806
1: 0.0000404, 0.0014309, -0.0000558, 0.0014004, -0.0010339, 0.0011476
2: 0.0141971, 0.0162795, 0.0142427, 0.0164236, -0.0017132, 0.0015376
3: 0.0000487, 0.0016146, 0.0000830, 0.0017229, -0.0012858, 0.0011515
4: -0.0043347, -0.0028903, -0.0043031, -0.0027904, -0.0012052, 0.0011029
5: 0.0079868, 0.0095498, 0.0080210, 0.0096580, -0.0012832, 0.0011490
6: 0.0093291, 0.0099189, 0.0092883, 0.0099060, -0.0005637, 0.0005454
7: -0.0191311, -0.0157379, -0.0193658, -0.0158122, -0.0024463, 0.0027558
8: 0.9689779, 0.9786999, 0.9683055, 0.9784872, -0.0071895, 0.0080038
9: 0.0039133, 0.0067705, 0.0039758, 0.0069682, -0.0023305, 0.0020750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051641, upper bound: 0.0055135
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051643, upper bound: 0.0055534
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002866, 0.0000311, -0.0002871, 0.0000859, -0.0002794, 0.0002309
1: -0.0000214, 0.0013692, -0.0000237, 0.0014533, -0.0010849, 0.0010384
2: 0.0142894, 0.0163721, 0.0141636, 0.0163754, -0.0015455, 0.0016208
3: 0.0001181, 0.0016842, 0.0000235, 0.0016867, -0.0011577, 0.0012170
4: -0.0042707, -0.0028261, -0.0043580, -0.0028238, -0.0011039, 0.0011367
5: 0.0080561, 0.0096193, 0.0079616, 0.0096218, -0.0011552, 0.0012146
6: 0.0093029, 0.0098928, 0.0093019, 0.0099284, -0.0005047, 0.0005714
7: -0.0192819, -0.0158884, -0.0192874, -0.0156833, -0.0026174, 0.0024563
8: 0.9685459, 0.9782689, 0.9685302, 0.9788564, -0.0075709, 0.0072251
9: 0.0040399, 0.0068975, 0.0038672, 0.0069021, -0.0020854, 0.0022105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049594, upper bound: 0.0050813
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049200, upper bound: 0.0050683
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002693, 0.0000731, -0.0002871, 0.0000923, -0.0002733, 0.0002688
1: 0.0000596, 0.0014336, -0.0000240, 0.0014631, -0.0010653, 0.0011071
2: 0.0141930, 0.0162507, 0.0141489, 0.0163760, -0.0016462, 0.0015837
3: 0.0000456, 0.0015930, 0.0000125, 0.0016872, -0.0012327, 0.0011856
4: -0.0043375, -0.0029103, -0.0043681, -0.0028234, -0.0011789, 0.0011383
5: 0.0079837, 0.0095282, 0.0079506, 0.0096222, -0.0012299, 0.0011829
6: 0.0093372, 0.0099201, 0.0093017, 0.0099326, -0.0005932, 0.0006167
7: -0.0190842, -0.0157313, -0.0192883, -0.0156594, -0.0025148, 0.0026106
8: 0.9691125, 0.9787189, 0.9685276, 0.9789250, -0.0074055, 0.0076975
9: 0.0039077, 0.0067310, 0.0038471, 0.0069029, -0.0022174, 0.0021344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049124, upper bound: 0.0052875
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049124, upper bound: 0.0052875
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0002902, 0.0000275, -0.0002871, 0.0000859, -0.0002802, 0.0002245
1: -0.0000385, 0.0013637, -0.0000237, 0.0014533, -0.0010939, 0.0010210
2: 0.0142977, 0.0163976, 0.0141636, 0.0163754, -0.0015185, 0.0016348
3: 0.0001243, 0.0017034, 0.0000235, 0.0016867, -0.0011371, 0.0012277
4: -0.0042649, -0.0028084, -0.0043580, -0.0028238, -0.0010866, 0.0011451
5: 0.0080623, 0.0096384, 0.0079616, 0.0096218, -0.0011345, 0.0012253
6: 0.0092956, 0.0098904, 0.0093019, 0.0099284, -0.0005016, 0.0005680
7: -0.0193235, -0.0159018, -0.0192874, -0.0156833, -0.0026422, 0.0024092
8: 0.9684270, 0.9782303, 0.9685302, 0.9788564, -0.0076358, 0.0071002
9: 0.0040513, 0.0069325, 0.0038672, 0.0069021, -0.0020455, 0.0022308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049468, upper bound: 0.0050733
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049141, upper bound: 0.0050518
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002734, 0.0000713, -0.0002871, 0.0000923, -0.0002735, 0.0002649
1: 0.0000404, 0.0014309, -0.0000240, 0.0014631, -0.0010732, 0.0010933
2: 0.0141971, 0.0162795, 0.0141489, 0.0163760, -0.0016265, 0.0015960
3: 0.0000487, 0.0016146, 0.0000125, 0.0016872, -0.0012180, 0.0011951
4: -0.0043347, -0.0028903, -0.0043681, -0.0028234, -0.0011645, 0.0011448
5: 0.0079868, 0.0095498, 0.0079506, 0.0096222, -0.0012153, 0.0011925
6: 0.0093291, 0.0099189, 0.0093017, 0.0099326, -0.0005906, 0.0006172
7: -0.0191311, -0.0157379, -0.0192883, -0.0156594, -0.0025343, 0.0025824
8: 0.9689779, 0.9786999, 0.9685276, 0.9789250, -0.0074624, 0.0076046
9: 0.0039133, 0.0067705, 0.0038471, 0.0069029, -0.0021930, 0.0021509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048993, upper bound: 0.0053184
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048993, upper bound: 0.0057518
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002757, 0.0000278, -0.0002235, 0.0002365
1: -0.0000339, 0.0013996, 0.0000297, 0.0013642, -0.0009971, 0.0009830
2: 0.0142439, 0.0163907, 0.0142969, 0.0162955, -0.0014688, 0.0014881
3: 0.0000839, 0.0016982, 0.0001238, 0.0016266, -0.0011029, 0.0011167
4: -0.0043022, -0.0028132, -0.0042655, -0.0028792, -0.0010291, 0.0010485
5: 0.0080219, 0.0096333, 0.0080617, 0.0095618, -0.0011008, 0.0011144
6: 0.0092976, 0.0099057, 0.0093245, 0.0098906, -0.0004787, 0.0004505
7: -0.0193123, -0.0158142, -0.0191571, -0.0159006, -0.0023916, 0.0023727
8: 0.9684589, 0.9784814, 0.9689034, 0.9782339, -0.0069525, 0.0068603
9: 0.0039775, 0.0069231, 0.0040502, 0.0067924, -0.0020034, 0.0020231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0041795, upper bound: 0.0034750
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0041795, upper bound: 0.0034750
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002820, 0.0000838, -0.0002757, 0.0000360, -0.0002356, 0.0002658
1: -0.0000001, 0.0014500, 0.0000295, 0.0013767, -0.0010504, 0.0010313
2: 0.0141685, 0.0163401, 0.0142782, 0.0162958, -0.0015400, 0.0015633
3: 0.0000272, 0.0016602, 0.0001097, 0.0016268, -0.0011559, 0.0011711
4: -0.0043545, -0.0028483, -0.0042785, -0.0028790, -0.0010830, 0.0011145
5: 0.0079653, 0.0095953, 0.0080477, 0.0095620, -0.0011536, 0.0011685
6: 0.0093119, 0.0099270, 0.0093245, 0.0098959, -0.0005716, 0.0004901
7: -0.0192298, -0.0156913, -0.0191576, -0.0158701, -0.0024895, 0.0024813
8: 0.9686952, 0.9788334, 0.9689021, 0.9783213, -0.0073086, 0.0071942
9: 0.0038740, 0.0068536, 0.0040245, 0.0067928, -0.0020967, 0.0021114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036998, upper bound: 0.0042891
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0036998, upper bound: 0.0055011
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000427, -0.0002794, 0.0000343, -0.0002358, 0.0002287
1: -0.0000336, 0.0013870, 0.0000120, 0.0013742, -0.0010321, 0.0009736
2: 0.0142627, 0.0163903, 0.0142819, 0.0163219, -0.0014535, 0.0015415
3: 0.0000981, 0.0016979, 0.0001125, 0.0016465, -0.0010910, 0.0011573
4: -0.0042892, -0.0028135, -0.0042759, -0.0028609, -0.0010221, 0.0010820
5: 0.0080361, 0.0096330, 0.0080505, 0.0095817, -0.0010888, 0.0011550
6: 0.0092977, 0.0099003, 0.0093171, 0.0098949, -0.0004790, 0.0004578
7: -0.0193116, -0.0158449, -0.0192002, -0.0158761, -0.0024857, 0.0023398
8: 0.9684608, 0.9783935, 0.9687800, 0.9783039, -0.0072008, 0.0067905
9: 0.0040033, 0.0069225, 0.0040296, 0.0068287, -0.0019780, 0.0021003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053425, upper bound: 0.0051641
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053935, upper bound: 0.0051643
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002820, 0.0000838, -0.0002794, 0.0000343, -0.0002357, 0.0002708
1: -0.0000001, 0.0014500, 0.0000120, 0.0013742, -0.0010537, 0.0010549
2: 0.0141685, 0.0163401, 0.0142819, 0.0163219, -0.0015752, 0.0015682
3: 0.0000272, 0.0016602, 0.0001125, 0.0016465, -0.0011825, 0.0011748
4: -0.0043545, -0.0028483, -0.0042759, -0.0028609, -0.0011066, 0.0011179
5: 0.0079653, 0.0095953, 0.0080505, 0.0095817, -0.0011801, 0.0011722
6: 0.0093119, 0.0099270, 0.0093171, 0.0098949, -0.0005730, 0.0004923
7: -0.0192298, -0.0156913, -0.0192002, -0.0158761, -0.0024975, 0.0025382
8: 0.9686952, 0.9788334, 0.9687800, 0.9783039, -0.0073316, 0.0073587
9: 0.0038740, 0.0068536, 0.0040296, 0.0068287, -0.0021450, 0.0021182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053935, upper bound: 0.0053878
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053935, upper bound: 0.0053985
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002895, 0.0000461, -0.0002754, 0.0000355, -0.0002330, 0.0002329
1: -0.0000350, 0.0013923, 0.0000310, 0.0013760, -0.0010114, 0.0009837
2: 0.0142549, 0.0163924, 0.0142793, 0.0162936, -0.0014686, 0.0015104
3: 0.0000922, 0.0016995, 0.0001105, 0.0016252, -0.0011022, 0.0011338
4: -0.0042946, -0.0028120, -0.0042777, -0.0028806, -0.0010324, 0.0010613
5: 0.0080302, 0.0096346, 0.0080485, 0.0095604, -0.0011000, 0.0011316
6: 0.0092971, 0.0099025, 0.0093251, 0.0098956, -0.0004729, 0.0004661
7: -0.0193150, -0.0158321, -0.0191540, -0.0158719, -0.0024345, 0.0023649
8: 0.9684511, 0.9784300, 0.9689125, 0.9783161, -0.0070554, 0.0068610
9: 0.0039926, 0.0069254, 0.0040261, 0.0067898, -0.0019988, 0.0020570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038514, upper bound: 0.0039350
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031552, upper bound: 0.0029618
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002904, 0.0000522, -0.0002753, 0.0000356, -0.0002357, 0.0002393
1: -0.0000391, 0.0014016, 0.0000315, 0.0013761, -0.0010334, 0.0009993
2: 0.0142410, 0.0163985, 0.0142791, 0.0162929, -0.0014920, 0.0015430
3: 0.0000817, 0.0017041, 0.0001104, 0.0016247, -0.0011198, 0.0011581
4: -0.0043042, -0.0028078, -0.0042778, -0.0028810, -0.0010487, 0.0010860
5: 0.0080197, 0.0096391, 0.0080483, 0.0095599, -0.0011175, 0.0011558
6: 0.0092954, 0.0099065, 0.0093253, 0.0098957, -0.0004968, 0.0004731
7: -0.0193249, -0.0158095, -0.0191529, -0.0158715, -0.0024836, 0.0024032
8: 0.9684227, 0.9784949, 0.9689157, 0.9783171, -0.0072082, 0.0069700
9: 0.0039735, 0.0069337, 0.0040257, 0.0067888, -0.0020307, 0.0020996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038192, upper bound: 0.0039031
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030602, upper bound: 0.0029071
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002895, 0.0000461, -0.0002791, 0.0000338, -0.0002267, 0.0002325
1: -0.0000350, 0.0013923, 0.0000136, 0.0013735, -0.0009870, 0.0009815
2: 0.0142549, 0.0163924, 0.0142830, 0.0163197, -0.0014655, 0.0014742
3: 0.0000922, 0.0016995, 0.0001133, 0.0016448, -0.0011001, 0.0011067
4: -0.0042946, -0.0028120, -0.0042751, -0.0028625, -0.0010297, 0.0010352
5: 0.0080302, 0.0096346, 0.0080513, 0.0095800, -0.0010979, 0.0011045
6: 0.0092971, 0.0099025, 0.0093177, 0.0098946, -0.0004662, 0.0004611
7: -0.0193150, -0.0158321, -0.0191965, -0.0158780, -0.0023765, 0.0023614
8: 0.9684511, 0.9784300, 0.9687906, 0.9782987, -0.0068864, 0.0068465
9: 0.0039926, 0.0069254, 0.0040312, 0.0068256, -0.0019957, 0.0020083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039858, upper bound: 0.0036003
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030803, upper bound: 0.0028311
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002904, 0.0000522, -0.0002790, 0.0000339, -0.0002310, 0.0002393
1: -0.0000391, 0.0014016, 0.0000140, 0.0013736, -0.0010178, 0.0009976
2: 0.0142410, 0.0163985, 0.0142828, 0.0163191, -0.0014898, 0.0015192
3: 0.0000817, 0.0017041, 0.0001132, 0.0016444, -0.0011183, 0.0011400
4: -0.0043042, -0.0028078, -0.0042752, -0.0028629, -0.0010466, 0.0010700
5: 0.0080197, 0.0096391, 0.0080511, 0.0095795, -0.0011161, 0.0011377
6: 0.0092954, 0.0099065, 0.0093179, 0.0098946, -0.0004932, 0.0004683
7: -0.0193249, -0.0158095, -0.0191956, -0.0158776, -0.0024437, 0.0024010
8: 0.9684227, 0.9784949, 0.9687933, 0.9782997, -0.0070976, 0.0069593
9: 0.0039735, 0.0069337, 0.0040309, 0.0068248, -0.0020291, 0.0020663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038101, upper bound: 0.0038771
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030122, upper bound: 0.0027955
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000842, -0.0002757, 0.0000360, -0.0002401, 0.0002666
1: -0.0000237, 0.0014506, 0.0000295, 0.0013767, -0.0010705, 0.0010344
2: 0.0141675, 0.0163755, 0.0142782, 0.0162958, -0.0015446, 0.0015938
3: 0.0000265, 0.0016868, 0.0001097, 0.0016268, -0.0011594, 0.0011943
4: -0.0043552, -0.0028238, -0.0042785, -0.0028790, -0.0010862, 0.0011362
5: 0.0079646, 0.0096219, 0.0080477, 0.0095620, -0.0011570, 0.0011917
6: 0.0093019, 0.0099273, 0.0093245, 0.0098959, -0.0005774, 0.0004914
7: -0.0192874, -0.0156898, -0.0191576, -0.0158701, -0.0025403, 0.0024888
8: 0.9685301, 0.9788378, 0.9689021, 0.9783213, -0.0074506, 0.0072157
9: 0.0038727, 0.0069022, 0.0040245, 0.0067928, -0.0021030, 0.0021544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034750, upper bound: 0.0041421
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0034750, upper bound: 0.0054989
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002871, 0.0000842, -0.0002794, 0.0000343, -0.0002365, 0.0002679
1: -0.0000237, 0.0014506, 0.0000120, 0.0013742, -0.0010605, 0.0010393
2: 0.0141675, 0.0163755, 0.0142819, 0.0163219, -0.0015519, 0.0015788
3: 0.0000265, 0.0016868, 0.0001125, 0.0016465, -0.0011650, 0.0011829
4: -0.0043552, -0.0028238, -0.0042759, -0.0028609, -0.0010905, 0.0011256
5: 0.0079646, 0.0096219, 0.0080505, 0.0095817, -0.0011627, 0.0011804
6: 0.0093019, 0.0099273, 0.0093171, 0.0098949, -0.0005750, 0.0004906
7: -0.0192874, -0.0156898, -0.0192002, -0.0158761, -0.0025142, 0.0025026
8: 0.9685301, 0.9788378, 0.9687800, 0.9783039, -0.0073803, 0.0072499
9: 0.0038727, 0.0069022, 0.0040296, 0.0068287, -0.0021144, 0.0021330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053609, upper bound: 0.0054065
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053625, upper bound: 0.0054102
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000445, -0.0002866, 0.0000311, -0.0002356, 0.0002483
1: -0.0000336, 0.0013899, -0.0000214, 0.0013692, -0.0010381, 0.0010588
2: 0.0142585, 0.0163903, 0.0142894, 0.0163721, -0.0015819, 0.0015502
3: 0.0000949, 0.0016979, 0.0001181, 0.0016842, -0.0011879, 0.0011636
4: -0.0042921, -0.0028135, -0.0042707, -0.0028261, -0.0011096, 0.0010890
5: 0.0080329, 0.0096330, 0.0080561, 0.0096193, -0.0011855, 0.0011613
6: 0.0092977, 0.0099015, 0.0093029, 0.0098928, -0.0004849, 0.0004860
7: -0.0193116, -0.0158380, -0.0192819, -0.0158884, -0.0024979, 0.0025547
8: 0.9684608, 0.9784132, 0.9685459, 0.9782689, -0.0072417, 0.0073891
9: 0.0039975, 0.0069225, 0.0040399, 0.0068975, -0.0021573, 0.0021111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046622, upper bound: 0.0041017
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0046008, upper bound: 0.0039842
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0002892, 0.0000509, -0.0002693, 0.0000731, -0.0002740, 0.0002401
1: -0.0000339, 0.0013996, 0.0000596, 0.0014336, -0.0011039, 0.0010263
2: 0.0142439, 0.0163907, 0.0141930, 0.0162507, -0.0015265, 0.0016482
3: 0.0000839, 0.0016982, 0.0000456, 0.0015930, -0.0011432, 0.0012370
4: -0.0043022, -0.0028132, -0.0043375, -0.0029103, -0.0010943, 0.0011595
5: 0.0080219, 0.0096333, 0.0079837, 0.0095282, -0.0011407, 0.0012345
6: 0.0092976, 0.0099057, 0.0093372, 0.0099201, -0.0005241, 0.0005665
7: -0.0193123, -0.0158142, -0.0190842, -0.0157313, -0.0026524, 0.0024285
8: 0.9684589, 0.9784814, 0.9691125, 0.9787189, -0.0076998, 0.0071371
9: 0.0039775, 0.0069231, 0.0039077, 0.0067310, -0.0020605, 0.0022427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.89 + 598.02 = 600.91 seconds
