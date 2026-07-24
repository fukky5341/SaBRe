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
Threshold: 0.000418


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002940, 0.0002940)
1: (-0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007461, 0.0007461)
2: (0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004629, 0.0004629)
3: (0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008643, 0.0008643)
4: (-0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007589, 0.0007589)
5: (0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002874, 0.0002874)
6: (0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010969, 0.0010969)
7: (0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007675, 0.0007675)
8: (-0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0008229, 0.0008229)
9: (-0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005436, 0.0005436)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.81 + 1.37 = 3.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0004672, upper bound: 0.0004672

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 91

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004543, upper bound: 0.0004543
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004543, upper bound: 0.0004543
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.06 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.06
Output dim: 7, lower bound: -0.0004543, upper bound: 0.0004543
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.06
Output dim: 7, lower bound: -0.0004543, upper bound: 0.0004543

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002924, 0.0002931
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007420, 0.0007437
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004603, 0.0004614
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008616, 0.0008595
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007547, 0.0007565
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002859, 0.0002865
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010935, 0.0010909
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007652, 0.0007633
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0008204, 0.0008184
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005406, 0.0005419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004455, upper bound: 0.0004444
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004444, upper bound: 0.0004455
time: 0.59 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002931, 0.0002924
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007437, 0.0007420
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004614, 0.0004603
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008595, 0.0008616
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007565, 0.0007547
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002865, 0.0002859
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010909, 0.0010935
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007633, 0.0007652
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0008184, 0.0008204
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005419, 0.0005406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004394, upper bound: 0.0004371
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004371, upper bound: 0.0004394
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 7, lower bound: -0.0004455, upper bound: 0.0004444
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 7, lower bound: -0.0004444, upper bound: 0.0004455
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 7, lower bound: -0.0004394, upper bound: 0.0004371
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 7, lower bound: -0.0004371, upper bound: 0.0004394

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002848, 0.0002877
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007226, 0.0007300
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004483, 0.0004529
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008457, 0.0008371
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007350, 0.0007426
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002784, 0.0002813
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010733, 0.0010624
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007511, 0.0007434
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0008052, 0.0007971
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005265, 0.0005319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 91

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004433, upper bound: 0.0004430
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004441, upper bound: 0.0004420
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002872, 0.0002855
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007288, 0.0007244
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004522, 0.0004494
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008392, 0.0008443
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007413, 0.0007368
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002808, 0.0002791
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010650, 0.0010715
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007453, 0.0007498
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007990, 0.0008039
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005310, 0.0005278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004286
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004256, upper bound: 0.0004300
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002833, 0.0002857
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007189, 0.0007251
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004460, 0.0004498
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008400, 0.0008328
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007313, 0.0007375
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002770, 0.0002794
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010660, 0.0010570
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007460, 0.0007396
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007998, 0.0007930
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005238, 0.0005283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004282, upper bound: 0.0004127
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004146, upper bound: 0.0004258
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002861, 0.0002826
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007261, 0.0007171
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004505, 0.0004449
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008308, 0.0008412
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007386, 0.0007295
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002798, 0.0002763
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010544, 0.0010676
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007378, 0.0007471
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007910, 0.0008010
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005291, 0.0005225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004347, upper bound: 0.0004381
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004358, upper bound: 0.0004372
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0004433, upper bound: 0.0004430
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0004441, upper bound: 0.0004420
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0004298, upper bound: 0.0004286
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0004256, upper bound: 0.0004300
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0004282, upper bound: 0.0004127
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0004146, upper bound: 0.0004258
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0004347, upper bound: 0.0004381
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0004358, upper bound: 0.0004372

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002807, 0.0002829
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007124, 0.0007180
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004420, 0.0004454
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008318, 0.0008253
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007246, 0.0007303
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002745, 0.0002766
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010556, 0.0010474
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007387, 0.0007329
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007920, 0.0007858
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005191, 0.0005231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004170, upper bound: 0.0004179
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004181, upper bound: 0.0004167
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002803, 0.0002836
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007113, 0.0007198
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004413, 0.0004466
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008338, 0.0008240
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007235, 0.0007322
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002740, 0.0002773
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010583, 0.0010458
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007405, 0.0007318
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007940, 0.0007846
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005183, 0.0005244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004440, upper bound: 0.0004297
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004352, upper bound: 0.0004418
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002771, 0.0002782
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007031, 0.0007059
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004362, 0.0004379
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008177, 0.0008145
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007152, 0.0007180
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002709, 0.0002720
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010378, 0.0010337
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007262, 0.0007233
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007786, 0.0007755
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005123, 0.0005143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004184, upper bound: 0.0004023
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004051, upper bound: 0.0004172
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002795, 0.0002753
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007093, 0.0006987
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004400, 0.0004335
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008094, 0.0008216
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007214, 0.0007107
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002733, 0.0002692
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010272, 0.0010428
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007188, 0.0007297
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007706, 0.0007823
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005168, 0.0005091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004141, upper bound: 0.0004049
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004029, upper bound: 0.0004190
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002653, 0.0002703
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006733, 0.0006859
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004177, 0.0004255
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007946, 0.0007800
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006849, 0.0006977
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002594, 0.0002643
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010084, 0.0009899
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007056, 0.0006927
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007565, 0.0007427
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004906, 0.0004997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004190, upper bound: 0.0004029
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004184, upper bound: 0.0004029
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002674, 0.0002678
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006786, 0.0006795
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004210, 0.0004215
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007871, 0.0007862
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006903, 0.0006911
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002615, 0.0002618
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009990, 0.0009978
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006990, 0.0006982
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007495, 0.0007486
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004945, 0.0004951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0002777, upper bound: 0.0002820
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0002777, upper bound: 0.0002820
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002821, 0.0002782
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007160, 0.0007059
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004442, 0.0004379
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008177, 0.0008294
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007283, 0.0007180
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002759, 0.0002720
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010378, 0.0010527
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007262, 0.0007366
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007786, 0.0007897
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005217, 0.0005143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004230, upper bound: 0.0004138
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004270
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002816, 0.0002786
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007146, 0.0007070
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004434, 0.0004386
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008190, 0.0008279
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007269, 0.0007191
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002753, 0.0002724
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010394, 0.0010507
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007273, 0.0007352
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007798, 0.0007882
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005207, 0.0005151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0002895, upper bound: 0.0002981
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0002895, upper bound: 0.0002981
time: 0.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004170, upper bound: 0.0004179
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004181, upper bound: 0.0004167
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004440, upper bound: 0.0004297
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004352, upper bound: 0.0004418
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004184, upper bound: 0.0004023
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004051, upper bound: 0.0004172
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004141, upper bound: 0.0004049
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004029, upper bound: 0.0004190
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004190, upper bound: 0.0004029
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004184, upper bound: 0.0004029
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0002777, upper bound: 0.0002820
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0002777, upper bound: 0.0002820
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004230, upper bound: 0.0004138
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0004077, upper bound: 0.0004270
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0002895, upper bound: 0.0002981
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 7, lower bound: -0.0002895, upper bound: 0.0002981

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002774, 0.0002800
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007039, 0.0007105
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004367, 0.0004408
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008231, 0.0008154
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007160, 0.0007227
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002712, 0.0002737
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010446, 0.0010349
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007310, 0.0007241
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007837, 0.0007764
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005129, 0.0005177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004180, upper bound: 0.0004028
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004095, upper bound: 0.0004165
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002803, 0.0002840
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007113, 0.0007207
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004413, 0.0004471
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008349, 0.0008240
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007235, 0.0007331
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002740, 0.0002777
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010596, 0.0010458
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007415, 0.0007318
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007950, 0.0007846
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005183, 0.0005251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004330, upper bound: 0.0004064
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004190, upper bound: 0.0004191
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002807, 0.0002836
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0007124, 0.0007198
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004420, 0.0004466
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0008338, 0.0008253
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0007246, 0.0007321
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002745, 0.0002773
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0010582, 0.0010474
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0007405, 0.0007329
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007939, 0.0007858
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0005191, 0.0005244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004236, upper bound: 0.0004161
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004092, upper bound: 0.0004299
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002581, 0.0002616
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006549, 0.0006639
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004063, 0.0004119
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007690, 0.0007587
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006662, 0.0006753
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002523, 0.0002558
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009760, 0.0009629
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006830, 0.0006738
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007323, 0.0007224
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004772, 0.0004837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004160, upper bound: 0.0004010
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004172, upper bound: 0.0003989
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002631, 0.0002563
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006675, 0.0006505
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004141, 0.0004036
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007536, 0.0007733
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006790, 0.0006617
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002572, 0.0002506
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009564, 0.0009814
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006692, 0.0006868
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007175, 0.0007363
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004864, 0.0004740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0003781, upper bound: 0.0003942
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0003793, upper bound: 0.0003933
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002563, 0.0002631
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006505, 0.0006675
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004036, 0.0004141
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007733, 0.0007536
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006617, 0.0006790
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002506, 0.0002572
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009814, 0.0009564
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006868, 0.0006692
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007363, 0.0007175
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004740, 0.0004864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0003933, upper bound: 0.0003793
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0003781
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002586, 0.0002613
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006561, 0.0006631
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004071, 0.0004114
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007681, 0.0007601
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006674, 0.0006745
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002528, 0.0002555
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009749, 0.0009647
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006822, 0.0006750
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007314, 0.0007237
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004781, 0.0004831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0002759, upper bound: 0.0002630
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0002759, upper bound: 0.0002630
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002655, 0.0002642
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006738, 0.0006705
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004180, 0.0004160
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007768, 0.0007805
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006853, 0.0006820
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002596, 0.0002583
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009858, 0.0009906
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006898, 0.0006932
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007396, 0.0007432
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004909, 0.0004886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0002807, upper bound: 0.0002710
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0002807, upper bound: 0.0002710
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002679, 0.0002615
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006799, 0.0006637
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004218, 0.0004117
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007688, 0.0007876
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006916, 0.0006751
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002619, 0.0002557
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009758, 0.0009996
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006828, 0.0006995
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007321, 0.0007499
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004954, 0.0004836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004076, upper bound: 0.0004164
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0003998, upper bound: 0.0004268
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0004180, upper bound: 0.0004028
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0004095, upper bound: 0.0004165
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0004330, upper bound: 0.0004064
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0004190, upper bound: 0.0004191
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0004236, upper bound: 0.0004161
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0004092, upper bound: 0.0004299
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0004160, upper bound: 0.0004010
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0004172, upper bound: 0.0003989
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0003781, upper bound: 0.0003942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0003793, upper bound: 0.0003933
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0003933, upper bound: 0.0003793
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0003942, upper bound: 0.0003781
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0002759, upper bound: 0.0002630
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0002759, upper bound: 0.0002630
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0002807, upper bound: 0.0002710
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0002807, upper bound: 0.0002710
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0004076, upper bound: 0.0004164
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 7, lower bound: -0.0003998, upper bound: 0.0004268

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002611, 0.0002672
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006627, 0.0006781
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004111, 0.0004207
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007855, 0.0007677
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006741, 0.0006897
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002553, 0.0002612
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009969, 0.0009743
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006976, 0.0006818
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007479, 0.0007310
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004828, 0.0004940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004067, upper bound: 0.0003826
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004083, upper bound: 0.0003743
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002638, 0.0002649
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006695, 0.0006721
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004154, 0.0004170
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007786, 0.0007756
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006810, 0.0006836
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002580, 0.0002589
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009881, 0.0009844
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006915, 0.0006888
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007414, 0.0007385
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004878, 0.0004897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004037, upper bound: 0.0004014
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004014, upper bound: 0.0004041
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002616, 0.0002669
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006638, 0.0006772
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004118, 0.0004201
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007845, 0.0007690
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006752, 0.0006888
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002557, 0.0002609
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009956, 0.0009759
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006967, 0.0006829
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007470, 0.0007322
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004836, 0.0004934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0002790, upper bound: 0.0002780
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0002790, upper bound: 0.0002780
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002642, 0.0002645
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006704, 0.0006712
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004159, 0.0004164
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007775, 0.0007766
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006819, 0.0006827
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002583, 0.0002586
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009868, 0.0009856
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006905, 0.0006897
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007403, 0.0007395
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004885, 0.0004890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0003813, upper bound: 0.0004050
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0003853, upper bound: 0.0004041
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011115, -0.0006537, -0.0011115, -0.0006537, -0.0002687, 0.0002620
1: -0.0071311, -0.0059695, -0.0071311, -0.0059695, -0.0006819, 0.0006648
2: 0.0306059, 0.0313265, 0.0306059, 0.0313265, -0.0004231, 0.0004125
3: 0.0010679, 0.0024136, 0.0010679, 0.0024136, -0.0007701, 0.0007900
4: -0.0061466, -0.0049650, -0.0061466, -0.0049650, -0.0006936, 0.0006762
5: 0.0114100, 0.0118576, 0.0114100, 0.0118576, -0.0002627, 0.0002561
6: 0.0017435, 0.0034513, 0.0017435, 0.0034513, -0.0009774, 0.0010026
7: 0.9792793, 0.9804743, 0.9792793, 0.9804743, -0.0006840, 0.0007016
8: -0.0087801, -0.0074988, -0.0087801, -0.0074988, -0.0007333, 0.0007522
9: -0.0000462, 0.0008002, -0.0000462, 0.0008002, -0.0004969, 0.0004844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0003911, upper bound: 0.0004170
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0003892, upper bound: 0.0004176
time: 0.60 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 7, lower bound: -0.0004067, upper bound: 0.0003826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 7, lower bound: -0.0004083, upper bound: 0.0003743
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 7, lower bound: -0.0004037, upper bound: 0.0004014
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 7, lower bound: -0.0004014, upper bound: 0.0004041
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 7, lower bound: -0.0002790, upper bound: 0.0002780
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 7, lower bound: -0.0002790, upper bound: 0.0002780
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 7, lower bound: -0.0003813, upper bound: 0.0004050
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 7, lower bound: -0.0003853, upper bound: 0.0004041
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 7, lower bound: -0.0003911, upper bound: 0.0004170
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 7, lower bound: -0.0003892, upper bound: 0.0004176

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.18 + 82.30 = 85.48 seconds
