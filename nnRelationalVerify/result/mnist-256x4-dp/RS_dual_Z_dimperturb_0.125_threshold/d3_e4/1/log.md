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
0: (-0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003917, 0.0003917)
1: (-0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009940, 0.0009940)
2: (0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0006167, 0.0006167)
3: (0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0011515, 0.0011515)
4: (-0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0010111, 0.0010111)
5: (0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003830, 0.0003830)
6: (0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0014614, 0.0014614)
7: (0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0010226, 0.0010226)
8: (-0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010964, 0.0010964)
9: (-0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0007242, 0.0007242)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.79 + 1.38 = 3.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0006151, upper bound: 0.0006151

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0006141, upper bound: 0.0005813
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005813, upper bound: 0.0006141
time: 0.50 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.21 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 7, lower bound: -0.0006141, upper bound: 0.0005813
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 7, lower bound: -0.0005813, upper bound: 0.0006141

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003900, 0.0003907
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009897, 0.0009915
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0006140, 0.0006151
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0011486, 0.0011466
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0010067, 0.0010085
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003813, 0.0003820
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0014577, 0.0014551
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0010200, 0.0010182
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010936, 0.0010917
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0007211, 0.0007224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0006002, upper bound: 0.0005655
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005990, upper bound: 0.0005655
time: 0.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003907, 0.0003900
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009915, 0.0009897
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0006151, 0.0006140
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0011466, 0.0011486
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0010085, 0.0010067
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003820, 0.0003813
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0014551, 0.0014577
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0010182, 0.0010200
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010917, 0.0010936
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0007224, 0.0007211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005655, upper bound: 0.0005990
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005655, upper bound: 0.0006002
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 7, lower bound: -0.0006002, upper bound: 0.0005655
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 7, lower bound: -0.0005990, upper bound: 0.0005655
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 7, lower bound: -0.0005655, upper bound: 0.0005990
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 7, lower bound: -0.0005655, upper bound: 0.0006002

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003788, 0.0003810
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009613, 0.0009669
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005964, 0.0005999
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0011201, 0.0011137
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009778, 0.0009835
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003704, 0.0003725
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0014216, 0.0014134
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009948, 0.0009890
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010666, 0.0010604
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0007004, 0.0007045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005910, upper bound: 0.0005451
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005864, upper bound: 0.0005582
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003803, 0.0003795
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009650, 0.0009631
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005987, 0.0005975
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0011157, 0.0011179
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009815, 0.0009796
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003718, 0.0003710
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0014159, 0.0014187
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009908, 0.0009927
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010623, 0.0010644
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0007031, 0.0007017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005901, upper bound: 0.0005459
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005823, upper bound: 0.0005580
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003795, 0.0003803
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009631, 0.0009650
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005975, 0.0005987
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0011179, 0.0011157
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009796, 0.0009815
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003710, 0.0003718
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0014187, 0.0014159
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009927, 0.0009908
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010644, 0.0010623
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0007017, 0.0007031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005580, upper bound: 0.0005823
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005460, upper bound: 0.0005901
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003810, 0.0003788
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009669, 0.0009613
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005999, 0.0005964
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0011137, 0.0011201
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009835, 0.0009778
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003725, 0.0003704
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0014134, 0.0014216
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009890, 0.0009948
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010604, 0.0010666
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0007045, 0.0007004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005582, upper bound: 0.0005865
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005450, upper bound: 0.0005910
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 7, lower bound: -0.0005910, upper bound: 0.0005451
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 7, lower bound: -0.0005864, upper bound: 0.0005582
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 7, lower bound: -0.0005901, upper bound: 0.0005459
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 7, lower bound: -0.0005823, upper bound: 0.0005580
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 7, lower bound: -0.0005580, upper bound: 0.0005823
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 7, lower bound: -0.0005460, upper bound: 0.0005901
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 7, lower bound: -0.0005582, upper bound: 0.0005865
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 7, lower bound: -0.0005450, upper bound: 0.0005910

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003564, 0.0003596
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009044, 0.0009126
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005611, 0.0005662
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010572, 0.0010477
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009199, 0.0009283
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003485, 0.0003516
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013418, 0.0013297
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009389, 0.0009305
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010067, 0.0009976
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006590, 0.0006650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005711, upper bound: 0.0005277
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005724, upper bound: 0.0005277
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003582, 0.0003586
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009091, 0.0009100
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005640, 0.0005646
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010542, 0.0010532
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009247, 0.0009256
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003503, 0.0003506
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013379, 0.0013366
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009362, 0.0009353
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010038, 0.0010028
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006624, 0.0006630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005676, upper bound: 0.0005406
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005676, upper bound: 0.0005398
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003578, 0.0003589
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009080, 0.0009108
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005633, 0.0005650
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010551, 0.0010519
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009236, 0.0009264
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003498, 0.0003509
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013390, 0.0013350
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009370, 0.0009342
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010046, 0.0010016
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006616, 0.0006636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005705, upper bound: 0.0005287
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005715, upper bound: 0.0005285
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003590, 0.0003571
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009110, 0.0009061
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005652, 0.0005622
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010497, 0.0010553
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009266, 0.0009217
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003510, 0.0003491
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013322, 0.0013393
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009322, 0.0009372
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009995, 0.0010048
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006637, 0.0006602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005638, upper bound: 0.0005406
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005641, upper bound: 0.0005398
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003571, 0.0003590
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009061, 0.0009110
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005622, 0.0005652
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010553, 0.0010497
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009217, 0.0009266
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003491, 0.0003510
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013393, 0.0013322
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009372, 0.0009322
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010048, 0.0009995
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006602, 0.0006637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005398, upper bound: 0.0005642
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005406, upper bound: 0.0005639
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003589, 0.0003578
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009108, 0.0009080
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005650, 0.0005633
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010519, 0.0010551
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009264, 0.0009236
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003509, 0.0003498
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013350, 0.0013390
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009342, 0.0009370
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010016, 0.0010046
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006636, 0.0006616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005285, upper bound: 0.0005715
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005287, upper bound: 0.0005705
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003586, 0.0003582
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009100, 0.0009091
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005646, 0.0005640
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010532, 0.0010542
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009256, 0.0009247
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003506, 0.0003503
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013366, 0.0013379
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009353, 0.0009362
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0010028, 0.0010038
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006630, 0.0006624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005398, upper bound: 0.0005676
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005406, upper bound: 0.0005676
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003596, 0.0003564
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0009126, 0.0009044
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005662, 0.0005611
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010477, 0.0010572
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009283, 0.0009199
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003516, 0.0003485
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013297, 0.0013418
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009305, 0.0009389
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009976, 0.0010067
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006650, 0.0006590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005277, upper bound: 0.0005724
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005277, upper bound: 0.0005711
time: 0.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005711, upper bound: 0.0005277
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005724, upper bound: 0.0005277
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005676, upper bound: 0.0005406
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005676, upper bound: 0.0005398
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005705, upper bound: 0.0005287
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005715, upper bound: 0.0005285
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005638, upper bound: 0.0005406
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005641, upper bound: 0.0005398
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005398, upper bound: 0.0005642
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005406, upper bound: 0.0005639
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005285, upper bound: 0.0005715
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005287, upper bound: 0.0005705
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005398, upper bound: 0.0005676
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005406, upper bound: 0.0005676
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005277, upper bound: 0.0005724
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.05
Output dim: 7, lower bound: -0.0005277, upper bound: 0.0005711

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003518, 0.0003532
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008928, 0.0008962
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005539, 0.0005560
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010382, 0.0010343
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009082, 0.0009116
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003440, 0.0003453
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013176, 0.0013127
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009220, 0.0009185
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009885, 0.0009848
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006505, 0.0006530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005634, upper bound: 0.0005008
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005250, upper bound: 0.0005166
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003499, 0.0003546
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008880, 0.0008998
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005509, 0.0005582
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010423, 0.0010287
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009032, 0.0009152
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003421, 0.0003467
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013229, 0.0013055
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009257, 0.0009135
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009925, 0.0009795
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006470, 0.0006556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005646, upper bound: 0.0005005
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005262, upper bound: 0.0005164
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003534, 0.0003521
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008968, 0.0008936
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005563, 0.0005544
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010352, 0.0010388
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009121, 0.0009089
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003455, 0.0003443
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013138, 0.0013184
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009193, 0.0009226
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009856, 0.0009891
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006534, 0.0006511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005594, upper bound: 0.0005068
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005203, upper bound: 0.0005315
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003518, 0.0003535
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008927, 0.0008971
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005538, 0.0005566
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010393, 0.0010341
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009080, 0.0009125
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003439, 0.0003456
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013190, 0.0013124
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009230, 0.0009184
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009896, 0.0009846
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006504, 0.0006537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005594, upper bound: 0.0005063
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005226, upper bound: 0.0005306
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003527, 0.0003524
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008951, 0.0008943
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005554, 0.0005548
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010360, 0.0010370
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009105, 0.0009097
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003449, 0.0003446
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013149, 0.0013161
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009201, 0.0009209
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009865, 0.0009874
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006522, 0.0006516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005617, upper bound: 0.0005008
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005250, upper bound: 0.0005175
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003513, 0.0003541
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008916, 0.0008986
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005532, 0.0005575
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010410, 0.0010329
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009069, 0.0009140
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003435, 0.0003462
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013212, 0.0013108
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009245, 0.0009173
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009912, 0.0009835
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006496, 0.0006547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005630, upper bound: 0.0005006
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005257, upper bound: 0.0005171
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003539, 0.0003506
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008980, 0.0008897
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005571, 0.0005520
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010307, 0.0010403
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009134, 0.0009050
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003460, 0.0003428
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013081, 0.0013202
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009153, 0.0009238
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009814, 0.0009905
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006543, 0.0006483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005544, upper bound: 0.0005069
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005195, upper bound: 0.0005314
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003525, 0.0003526
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008945, 0.0008947
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005550, 0.0005551
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010364, 0.0010363
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009099, 0.0009100
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003446, 0.0003447
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013154, 0.0013152
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009204, 0.0009203
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009868, 0.0009867
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006518, 0.0006519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005554, upper bound: 0.0005064
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005202, upper bound: 0.0005306
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003526, 0.0003525
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008947, 0.0008945
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005551, 0.0005550
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010363, 0.0010364
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009100, 0.0009099
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003447, 0.0003446
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013152, 0.0013154
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009203, 0.0009204
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009867, 0.0009868
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006519, 0.0006518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005305, upper bound: 0.0005202
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005064, upper bound: 0.0005554
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003506, 0.0003539
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008897, 0.0008980
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005520, 0.0005571
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010403, 0.0010307
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009050, 0.0009134
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003428, 0.0003460
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013202, 0.0013081
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009238, 0.0009153
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009905, 0.0009814
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006483, 0.0006543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005314, upper bound: 0.0005195
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005069, upper bound: 0.0005544
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003541, 0.0003513
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008986, 0.0008916
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005575, 0.0005532
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010329, 0.0010410
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009140, 0.0009069
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003462, 0.0003435
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013108, 0.0013212
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009173, 0.0009245
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009835, 0.0009912
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006547, 0.0006496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005171, upper bound: 0.0005257
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005006, upper bound: 0.0005630
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003524, 0.0003527
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008943, 0.0008951
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005548, 0.0005554
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010370, 0.0010360
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009097, 0.0009105
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003446, 0.0003449
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013161, 0.0013149
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009209, 0.0009201
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009874, 0.0009865
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006516, 0.0006522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005175, upper bound: 0.0005250
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005008, upper bound: 0.0005617
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003535, 0.0003518
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008971, 0.0008927
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005566, 0.0005538
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010341, 0.0010393
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009125, 0.0009080
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003456, 0.0003439
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013124, 0.0013190
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009184, 0.0009230
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009846, 0.0009896
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006537, 0.0006504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005305, upper bound: 0.0005226
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005063, upper bound: 0.0005594
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003521, 0.0003534
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008936, 0.0008968
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005544, 0.0005563
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010388, 0.0010352
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009089, 0.0009121
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003443, 0.0003455
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013184, 0.0013138
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009226, 0.0009193
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009891, 0.0009856
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006511, 0.0006534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005314, upper bound: 0.0005203
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005068, upper bound: 0.0005594
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003546, 0.0003499
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008998, 0.0008880
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005582, 0.0005509
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010287, 0.0010423
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009152, 0.0009032
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003467, 0.0003421
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013055, 0.0013229
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009135, 0.0009257
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009795, 0.0009925
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006556, 0.0006470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005164, upper bound: 0.0005262
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005005, upper bound: 0.0005646
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003532, 0.0003518
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008962, 0.0008928
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005560, 0.0005539
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010343, 0.0010382
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0009116, 0.0009082
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003453, 0.0003440
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0013127, 0.0013176
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0009185, 0.0009220
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009848, 0.0009885
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006530, 0.0006505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005166, upper bound: 0.0005250
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005008, upper bound: 0.0005634
time: 0.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005634, upper bound: 0.0005008
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005250, upper bound: 0.0005166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005646, upper bound: 0.0005005
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005262, upper bound: 0.0005164
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005594, upper bound: 0.0005068
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005203, upper bound: 0.0005315
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005594, upper bound: 0.0005063
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005226, upper bound: 0.0005306
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005617, upper bound: 0.0005008
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005250, upper bound: 0.0005175
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005630, upper bound: 0.0005006
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005257, upper bound: 0.0005171
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005544, upper bound: 0.0005069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005195, upper bound: 0.0005314
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005554, upper bound: 0.0005064
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005202, upper bound: 0.0005306
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005305, upper bound: 0.0005202
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005064, upper bound: 0.0005554
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005314, upper bound: 0.0005195
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005069, upper bound: 0.0005544
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005171, upper bound: 0.0005257
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005006, upper bound: 0.0005630
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005175, upper bound: 0.0005250
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005008, upper bound: 0.0005617
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005305, upper bound: 0.0005226
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005063, upper bound: 0.0005594
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005314, upper bound: 0.0005203
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005068, upper bound: 0.0005594
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005164, upper bound: 0.0005262
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005005, upper bound: 0.0005646
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005166, upper bound: 0.0005250
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 7, lower bound: -0.0005008, upper bound: 0.0005634

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003342, 0.0003389
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008480, 0.0008601
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005261, 0.0005336
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009963, 0.0009824
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008626, 0.0008748
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003267, 0.0003314
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012645, 0.0012468
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008848, 0.0008725
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009487, 0.0009354
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006179, 0.0006267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005447, upper bound: 0.0004835
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005482, upper bound: 0.0004839
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003371, 0.0003355
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008554, 0.0008514
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005307, 0.0005282
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009863, 0.0009909
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008700, 0.0008660
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003296, 0.0003280
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012518, 0.0012576
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008759, 0.0008800
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009391, 0.0009435
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006232, 0.0006204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005091, upper bound: 0.0005006
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005101, upper bound: 0.0005005
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003323, 0.0003402
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008432, 0.0008632
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005231, 0.0005355
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0010000, 0.0009768
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008577, 0.0008780
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003249, 0.0003326
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012691, 0.0012397
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008881, 0.0008675
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009522, 0.0009301
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006144, 0.0006290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005456, upper bound: 0.0004835
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005495, upper bound: 0.0004838
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003355, 0.0003369
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008514, 0.0008550
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005282, 0.0005304
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009905, 0.0009863
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008660, 0.0008697
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003280, 0.0003294
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012570, 0.0012517
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008796, 0.0008759
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009431, 0.0009391
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006203, 0.0006230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005107, upper bound: 0.0005006
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005114, upper bound: 0.0005005
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003357, 0.0003380
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008520, 0.0008577
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005286, 0.0005321
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009936, 0.0009870
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008666, 0.0008724
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003282, 0.0003304
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012610, 0.0012526
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008824, 0.0008765
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009460, 0.0009398
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006208, 0.0006249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005403, upper bound: 0.0004890
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005442, upper bound: 0.0004897
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003390, 0.0003345
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008602, 0.0008488
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005337, 0.0005266
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009833, 0.0009965
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008750, 0.0008634
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003314, 0.0003270
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012479, 0.0012647
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008732, 0.0008850
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009362, 0.0009488
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006267, 0.0006184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005041, upper bound: 0.0005164
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005060, upper bound: 0.0005154
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003341, 0.0003387
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008479, 0.0008594
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005260, 0.0005332
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009956, 0.0009822
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008624, 0.0008741
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003267, 0.0003311
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012635, 0.0012466
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008841, 0.0008723
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009479, 0.0009352
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006178, 0.0006262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005410, upper bound: 0.0004886
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005443, upper bound: 0.0004893
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003374, 0.0003359
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008561, 0.0008524
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005311, 0.0005288
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009874, 0.0009917
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008708, 0.0008670
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003298, 0.0003284
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012532, 0.0012586
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008769, 0.0008807
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009402, 0.0009443
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006238, 0.0006210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005062, upper bound: 0.0005162
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005082, upper bound: 0.0005147
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003351, 0.0003380
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008504, 0.0008577
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005276, 0.0005321
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009936, 0.0009851
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008650, 0.0008725
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003276, 0.0003305
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012611, 0.0012502
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008824, 0.0008749
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009461, 0.0009380
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006196, 0.0006250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005434, upper bound: 0.0004845
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005470, upper bound: 0.0004845
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003379, 0.0003348
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008574, 0.0008495
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005319, 0.0005271
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009842, 0.0009932
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008721, 0.0008641
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003303, 0.0003273
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012490, 0.0012605
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008740, 0.0008821
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009371, 0.0009457
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006247, 0.0006190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005091, upper bound: 0.0005021
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005101, upper bound: 0.0005016
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003337, 0.0003397
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008468, 0.0008619
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005254, 0.0005348
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009985, 0.0009810
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008614, 0.0008767
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003263, 0.0003321
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012672, 0.0012450
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008868, 0.0008712
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009507, 0.0009341
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006170, 0.0006280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005452, upper bound: 0.0004844
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005483, upper bound: 0.0004844
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003374, 0.0003365
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008561, 0.0008538
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005311, 0.0005297
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009891, 0.0009917
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008708, 0.0008685
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003298, 0.0003290
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012553, 0.0012587
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008784, 0.0008807
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009418, 0.0009443
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006238, 0.0006221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005104, upper bound: 0.0005020
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005108, upper bound: 0.0005013
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003362, 0.0003362
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008532, 0.0008531
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005293, 0.0005293
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009883, 0.0009884
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008679, 0.0008677
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003287, 0.0003287
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012542, 0.0012544
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008777, 0.0008778
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009410, 0.0009411
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006217, 0.0006216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005377, upper bound: 0.0004898
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005393, upper bound: 0.0004900
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003395, 0.0003330
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008615, 0.0008449
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005345, 0.0005242
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009788, 0.0009980
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008762, 0.0008594
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003319, 0.0003255
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012422, 0.0012665
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008693, 0.0008863
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009320, 0.0009502
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006277, 0.0006156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005036, upper bound: 0.0005165
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005047, upper bound: 0.0005154
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003349, 0.0003378
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008497, 0.0008572
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005272, 0.0005318
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009930, 0.0009844
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008643, 0.0008719
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003274, 0.0003303
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012603, 0.0012493
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008819, 0.0008742
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009455, 0.0009373
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006191, 0.0006246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005387, upper bound: 0.0004897
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005397, upper bound: 0.0004898
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003383, 0.0003349
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008586, 0.0008499
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005327, 0.0005273
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009846, 0.0009946
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008733, 0.0008645
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003308, 0.0003274
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012495, 0.0012623
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008744, 0.0008833
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009374, 0.0009470
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006256, 0.0006192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005044, upper bound: 0.0005162
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0005147
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003349, 0.0003383
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008499, 0.0008586
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005273, 0.0005327
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009946, 0.0009846
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008645, 0.0008733
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003274, 0.0003308
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012623, 0.0012495
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008833, 0.0008744
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009470, 0.0009374
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006192, 0.0006256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005147, upper bound: 0.0005057
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005162, upper bound: 0.0005044
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003378, 0.0003349
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008572, 0.0008497
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005318, 0.0005272
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009844, 0.0009930
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008719, 0.0008643
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003303, 0.0003274
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012493, 0.0012603
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008742, 0.0008819
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009373, 0.0009455
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006246, 0.0006191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004898, upper bound: 0.0005397
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004897, upper bound: 0.0005387
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003330, 0.0003395
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008449, 0.0008615
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005242, 0.0005345
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009980, 0.0009788
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008594, 0.0008762
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003255, 0.0003319
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012665, 0.0012422
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008863, 0.0008693
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009502, 0.0009320
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006156, 0.0006277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005154, upper bound: 0.0005047
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005165, upper bound: 0.0005036
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003362, 0.0003362
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008531, 0.0008532
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005293, 0.0005293
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009884, 0.0009883
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008677, 0.0008679
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003287, 0.0003287
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012544, 0.0012542
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008778, 0.0008777
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009411, 0.0009410
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006216, 0.0006217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004901, upper bound: 0.0005393
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004898, upper bound: 0.0005377
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003365, 0.0003374
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008538, 0.0008561
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005297, 0.0005311
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009917, 0.0009891
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008685, 0.0008708
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003290, 0.0003298
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012587, 0.0012553
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008807, 0.0008784
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009443, 0.0009418
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006221, 0.0006238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005013, upper bound: 0.0005108
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005020, upper bound: 0.0005104
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003397, 0.0003337
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008619, 0.0008468
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005348, 0.0005254
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009810, 0.0009985
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008767, 0.0008614
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003321, 0.0003263
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012450, 0.0012672
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008712, 0.0008868
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009341, 0.0009507
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006280, 0.0006170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0005483
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0005452
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003348, 0.0003379
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008495, 0.0008574
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005271, 0.0005319
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009932, 0.0009842
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008641, 0.0008721
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003273, 0.0003303
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012605, 0.0012490
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008821, 0.0008740
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009457, 0.0009371
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006190, 0.0006247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005016, upper bound: 0.0005101
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005021, upper bound: 0.0005091
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003380, 0.0003351
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008577, 0.0008504
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005321, 0.0005276
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009851, 0.0009936
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008725, 0.0008650
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003305, 0.0003276
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012502, 0.0012611
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008749, 0.0008824
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009380, 0.0009461
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006250, 0.0006196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004845, upper bound: 0.0005470
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004845, upper bound: 0.0005434
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003359, 0.0003374
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008524, 0.0008561
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005288, 0.0005311
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009917, 0.0009874
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008670, 0.0008708
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003284, 0.0003298
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012586, 0.0012532
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008807, 0.0008769
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009443, 0.0009402
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006210, 0.0006238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005147, upper bound: 0.0005083
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005162, upper bound: 0.0005062
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003387, 0.0003341
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008594, 0.0008479
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005332, 0.0005260
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009822, 0.0009956
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008741, 0.0008624
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003311, 0.0003267
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012466, 0.0012635
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008723, 0.0008841
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009352, 0.0009479
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006262, 0.0006178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004893, upper bound: 0.0005443
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004886, upper bound: 0.0005410
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003345, 0.0003390
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008488, 0.0008602
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005266, 0.0005337
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009965, 0.0009833
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008634, 0.0008750
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003270, 0.0003314
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012647, 0.0012479
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008850, 0.0008732
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009488, 0.0009362
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006184, 0.0006267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005154, upper bound: 0.0005060
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005164, upper bound: 0.0005042
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003380, 0.0003357
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008577, 0.0008520
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005321, 0.0005286
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009870, 0.0009936
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008724, 0.0008666
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003304, 0.0003282
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012526, 0.0012610
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008765, 0.0008824
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009398, 0.0009460
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006249, 0.0006208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004896, upper bound: 0.0005442
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004890, upper bound: 0.0005403
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003369, 0.0003355
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008550, 0.0008514
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005304, 0.0005282
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009863, 0.0009905
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008697, 0.0008660
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003294, 0.0003280
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012517, 0.0012570
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008759, 0.0008796
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009391, 0.0009431
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006230, 0.0006203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005004, upper bound: 0.0005114
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005005, upper bound: 0.0005107
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003402, 0.0003323
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008632, 0.0008432
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005355, 0.0005231
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009768, 0.0010000
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008780, 0.0008577
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003326, 0.0003249
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012397, 0.0012691
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008675, 0.0008881
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009301, 0.0009522
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006290, 0.0006144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004838, upper bound: 0.0005495
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004835, upper bound: 0.0005456
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003355, 0.0003371
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008514, 0.0008554
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005282, 0.0005307
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009909, 0.0009863
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008660, 0.0008700
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003280, 0.0003296
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012576, 0.0012518
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008800, 0.0008759
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009435, 0.0009391
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006204, 0.0006232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005005, upper bound: 0.0005101
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005005, upper bound: 0.0005091
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003389, 0.0003342
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008601, 0.0008480
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005336, 0.0005261
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009824, 0.0009963
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008748, 0.0008626
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003314, 0.0003267
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012468, 0.0012645
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008725, 0.0008848
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009354, 0.0009487
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006267, 0.0006179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004839, upper bound: 0.0005482
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004835, upper bound: 0.0005447
time: 0.60 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005447, upper bound: 0.0004835
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005482, upper bound: 0.0004839
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005091, upper bound: 0.0005006
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005101, upper bound: 0.0005005
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005456, upper bound: 0.0004835
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005495, upper bound: 0.0004838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005107, upper bound: 0.0005006
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005114, upper bound: 0.0005005
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005403, upper bound: 0.0004890
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005442, upper bound: 0.0004897
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005041, upper bound: 0.0005164
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005060, upper bound: 0.0005154
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005410, upper bound: 0.0004886
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005443, upper bound: 0.0004893
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005062, upper bound: 0.0005162
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005082, upper bound: 0.0005147
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005434, upper bound: 0.0004845
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005470, upper bound: 0.0004845
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005091, upper bound: 0.0005021
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005101, upper bound: 0.0005016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005452, upper bound: 0.0004844
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005483, upper bound: 0.0004844
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005104, upper bound: 0.0005020
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005108, upper bound: 0.0005013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005377, upper bound: 0.0004898
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005393, upper bound: 0.0004900
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005036, upper bound: 0.0005165
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005047, upper bound: 0.0005154
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005387, upper bound: 0.0004897
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005397, upper bound: 0.0004898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005044, upper bound: 0.0005162
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0005147
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005147, upper bound: 0.0005057
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005162, upper bound: 0.0005044
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004898, upper bound: 0.0005397
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004897, upper bound: 0.0005387
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005154, upper bound: 0.0005047
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005165, upper bound: 0.0005036
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004901, upper bound: 0.0005393
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004898, upper bound: 0.0005377
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005013, upper bound: 0.0005108
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005020, upper bound: 0.0005104
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0005483
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0005452
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005016, upper bound: 0.0005101
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005021, upper bound: 0.0005091
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004845, upper bound: 0.0005470
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004845, upper bound: 0.0005434
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005147, upper bound: 0.0005083
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005162, upper bound: 0.0005062
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004893, upper bound: 0.0005443
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004886, upper bound: 0.0005410
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005154, upper bound: 0.0005060
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005164, upper bound: 0.0005042
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004896, upper bound: 0.0005442
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004890, upper bound: 0.0005403
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005004, upper bound: 0.0005114
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005005, upper bound: 0.0005107
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004838, upper bound: 0.0005495
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004835, upper bound: 0.0005456
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005005, upper bound: 0.0005101
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0005005, upper bound: 0.0005091
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004839, upper bound: 0.0005482
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 7, lower bound: -0.0004835, upper bound: 0.0005447

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003239, 0.0003288
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008221, 0.0008344
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005100, 0.0005177
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009666, 0.0009523
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008362, 0.0008487
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003167, 0.0003215
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012267, 0.0012086
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008584, 0.0008457
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009203, 0.0009068
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005990, 0.0006079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004152
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004152
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003241, 0.0003293
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008224, 0.0008356
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005102, 0.0005184
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009681, 0.0009527
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008365, 0.0008500
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003168, 0.0003220
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012286, 0.0012091
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008597, 0.0008460
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009217, 0.0009071
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005992, 0.0006089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004152
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004152
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003270, 0.0003254
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008299, 0.0008257
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005149, 0.0005123
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009566, 0.0009614
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008442, 0.0008399
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003197, 0.0003181
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012140, 0.0012201
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008495, 0.0008538
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009108, 0.0009154
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006047, 0.0006016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004380
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004380
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003270, 0.0003259
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008297, 0.0008271
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005147, 0.0005131
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009582, 0.0009612
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008439, 0.0008413
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003197, 0.0003187
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012160, 0.0012198
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008509, 0.0008536
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009123, 0.0009152
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006045, 0.0006026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004380
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004380
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003224, 0.0003300
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008183, 0.0008375
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005076, 0.0005196
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009703, 0.0009479
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008323, 0.0008519
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003153, 0.0003227
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012314, 0.0012030
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008617, 0.0008418
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009238, 0.0009026
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005962, 0.0006102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004152
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004152
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003222, 0.0003304
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008175, 0.0008385
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005072, 0.0005202
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009714, 0.0009471
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008316, 0.0008529
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003150, 0.0003231
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012328, 0.0012019
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008627, 0.0008411
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009249, 0.0009018
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005957, 0.0006109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004152
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004152
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003258, 0.0003268
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008268, 0.0008293
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005130, 0.0005145
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009607, 0.0009579
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008410, 0.0008435
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003186, 0.0003195
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012193, 0.0012156
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008532, 0.0008506
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009147, 0.0009120
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006024, 0.0006042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004380
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004380
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003254, 0.0003272
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008257, 0.0008302
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005123, 0.0005151
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009618, 0.0009566
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008399, 0.0008445
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003181, 0.0003199
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012206, 0.0012140
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008541, 0.0008495
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009158, 0.0009108
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006016, 0.0006049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004380
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004380
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003253, 0.0003279
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008255, 0.0008320
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005122, 0.0005162
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009638, 0.0009563
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008397, 0.0008463
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003181, 0.0003205
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012232, 0.0012137
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008559, 0.0008493
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009177, 0.0009106
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006015, 0.0006062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0004253
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0004253
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003256, 0.0003287
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008263, 0.0008341
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005126, 0.0005175
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009662, 0.0009572
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008405, 0.0008484
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003184, 0.0003213
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012263, 0.0012148
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008581, 0.0008501
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009200, 0.0009114
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006021, 0.0006077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004741, upper bound: 0.0004253
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004741, upper bound: 0.0004253
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003287, 0.0003244
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008342, 0.0008231
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005175, 0.0005107
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009535, 0.0009663
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008485, 0.0008372
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003214, 0.0003171
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012102, 0.0012264
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008468, 0.0008582
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009079, 0.0009201
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006078, 0.0005997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003289, 0.0003254
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008345, 0.0008257
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005177, 0.0005123
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009565, 0.0009667
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008488, 0.0008399
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003215, 0.0003181
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012140, 0.0012269
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008495, 0.0008585
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009108, 0.0009205
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006080, 0.0006016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003238, 0.0003285
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008217, 0.0008337
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005098, 0.0005172
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009658, 0.0009519
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008358, 0.0008480
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003166, 0.0003212
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012257, 0.0012081
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008577, 0.0008454
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009196, 0.0009064
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005987, 0.0006074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0004253
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0004253
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003240, 0.0003294
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008222, 0.0008359
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005101, 0.0005186
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009684, 0.0009525
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008363, 0.0008503
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003168, 0.0003221
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012290, 0.0012088
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008600, 0.0008459
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009220, 0.0009069
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005991, 0.0006091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004741, upper bound: 0.0004253
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004741, upper bound: 0.0004253
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003273, 0.0003258
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008305, 0.0008267
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005153, 0.0005129
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009577, 0.0009621
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008448, 0.0008409
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003200, 0.0003185
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012154, 0.0012211
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008505, 0.0008545
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009119, 0.0009161
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006051, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003272, 0.0003266
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008304, 0.0008288
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005152, 0.0005142
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009601, 0.0009620
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008447, 0.0008430
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003199, 0.0003193
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012185, 0.0012209
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008526, 0.0008543
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009142, 0.0009160
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006050, 0.0006039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003258, 0.0003279
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008267, 0.0008321
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005129, 0.0005162
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009639, 0.0009577
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008409, 0.0008463
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003185, 0.0003206
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012233, 0.0012154
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008560, 0.0008505
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009178, 0.0009118
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006023, 0.0006063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004156
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004156
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003250, 0.0003280
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008247, 0.0008323
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005116, 0.0005164
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009642, 0.0009554
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008388, 0.0008466
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003177, 0.0003207
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012237, 0.0012125
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008563, 0.0008484
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009181, 0.0009097
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006009, 0.0006064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004156
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004156
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003286, 0.0003247
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008338, 0.0008239
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005173, 0.0005111
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009544, 0.0009659
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008481, 0.0008380
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003212, 0.0003174
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012113, 0.0012259
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008476, 0.0008578
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009088, 0.0009197
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006075, 0.0006003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004402
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004402
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003277, 0.0003245
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008317, 0.0008235
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005160, 0.0005109
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009540, 0.0009635
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008460, 0.0008376
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003204, 0.0003173
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012107, 0.0012228
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008472, 0.0008556
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009083, 0.0009174
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006060, 0.0006000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004402
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004402
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003246, 0.0003295
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008237, 0.0008363
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005110, 0.0005188
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009688, 0.0009542
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008379, 0.0008506
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003174, 0.0003222
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012295, 0.0012110
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008603, 0.0008474
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009224, 0.0009086
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006002, 0.0006093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004156
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004156
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003236, 0.0003294
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008211, 0.0008360
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005094, 0.0005186
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009684, 0.0009513
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008352, 0.0008503
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003164, 0.0003221
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012291, 0.0012073
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008600, 0.0008448
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009221, 0.0009057
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005983, 0.0006091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004156
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004156
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003280, 0.0003263
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008324, 0.0008282
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005164, 0.0005138
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009594, 0.0009643
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008467, 0.0008424
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003207, 0.0003191
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012176, 0.0012238
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008520, 0.0008564
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009135, 0.0009182
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006065, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004402
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004402
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003272, 0.0003261
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008304, 0.0008274
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005152, 0.0005133
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009585, 0.0009620
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008447, 0.0008416
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003199, 0.0003188
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012165, 0.0012209
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008512, 0.0008543
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009127, 0.0009160
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006051, 0.0006029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004402
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004402
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003264, 0.0003261
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008283, 0.0008274
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005139, 0.0005133
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009585, 0.0009595
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008425, 0.0008416
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003191, 0.0003188
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012165, 0.0012177
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008512, 0.0008521
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009127, 0.0009136
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006035, 0.0006029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004251
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004251
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003261, 0.0003265
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008275, 0.0008286
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005134, 0.0005141
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009599, 0.0009587
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008417, 0.0008428
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003188, 0.0003192
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012182, 0.0012167
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008525, 0.0008514
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009140, 0.0009128
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006030, 0.0006037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004734, upper bound: 0.0004251
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004734, upper bound: 0.0004251
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003297, 0.0003228
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008366, 0.0008193
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005190, 0.0005083
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009491, 0.0009691
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008509, 0.0008333
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003223, 0.0003156
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012045, 0.0012300
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008428, 0.0008607
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009037, 0.0009228
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006095, 0.0005969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003294, 0.0003232
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008358, 0.0008201
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005185, 0.0005088
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009500, 0.0009682
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008501, 0.0008342
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003220, 0.0003160
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012057, 0.0012288
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008437, 0.0008598
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009046, 0.0009219
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006090, 0.0005975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003252, 0.0003277
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008252, 0.0008315
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005120, 0.0005159
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009633, 0.0009560
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008394, 0.0008458
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003179, 0.0003204
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012225, 0.0012132
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008555, 0.0008490
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009172, 0.0009102
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006013, 0.0006059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004251
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004251
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003247, 0.0003278
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008241, 0.0008319
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005113, 0.0005161
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009637, 0.0009546
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008382, 0.0008462
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003175, 0.0003205
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012231, 0.0012116
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008558, 0.0008478
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009176, 0.0009090
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006004, 0.0006061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004734, upper bound: 0.0004251
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004734, upper bound: 0.0004251
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003287, 0.0003248
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008341, 0.0008242
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005175, 0.0005113
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009548, 0.0009663
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008484, 0.0008384
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003214, 0.0003175
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012118, 0.0012263
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008479, 0.0008581
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009091, 0.0009200
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006077, 0.0006005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003282, 0.0003248
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008329, 0.0008241
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005167, 0.0005113
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009547, 0.0009648
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008472, 0.0008383
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003209, 0.0003175
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012117, 0.0012245
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008479, 0.0008569
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009090, 0.0009187
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006068, 0.0006005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003248, 0.0003282
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008241, 0.0008329
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005113, 0.0005167
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009648, 0.0009547
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008383, 0.0008472
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003175, 0.0003209
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012245, 0.0012117
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008569, 0.0008479
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009187, 0.0009090
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006005, 0.0006068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003248, 0.0003287
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008242, 0.0008341
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005113, 0.0005175
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009663, 0.0009548
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008384, 0.0008484
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003175, 0.0003214
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012263, 0.0012118
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008581, 0.0008479
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009200, 0.0009091
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006005, 0.0006077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003278, 0.0003247
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008319, 0.0008241
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005161, 0.0005113
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009546, 0.0009637
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008462, 0.0008382
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003205, 0.0003175
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012116, 0.0012231
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008478, 0.0008558
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009090, 0.0009176
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006061, 0.0006004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004734
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004734
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003277, 0.0003252
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008315, 0.0008252
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005159, 0.0005120
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009560, 0.0009633
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008458, 0.0008394
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003204, 0.0003179
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012132, 0.0012225
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008490, 0.0008555
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009102, 0.0009172
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006059, 0.0006013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004722
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004722
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003232, 0.0003294
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008201, 0.0008358
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005088, 0.0005185
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009682, 0.0009500
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008342, 0.0008501
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003160, 0.0003220
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012288, 0.0012057
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008598, 0.0008437
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009219, 0.0009046
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005975, 0.0006090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003228, 0.0003297
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008193, 0.0008366
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005083, 0.0005190
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009691, 0.0009491
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008333, 0.0008509
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003156, 0.0003223
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012300, 0.0012045
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008607, 0.0008428
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009228, 0.0009037
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005969, 0.0006095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003265, 0.0003261
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008286, 0.0008275
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005141, 0.0005134
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009587, 0.0009599
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008428, 0.0008417
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003192, 0.0003188
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012167, 0.0012182
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008514, 0.0008525
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009128, 0.0009140
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006037, 0.0006030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004734
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004734
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003261, 0.0003264
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008274, 0.0008283
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005133, 0.0005139
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009595, 0.0009585
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008416, 0.0008425
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003188, 0.0003191
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012177, 0.0012165
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008521, 0.0008512
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009136, 0.0009127
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006029, 0.0006035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004722
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004722
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003261, 0.0003272
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008274, 0.0008304
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005133, 0.0005152
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009620, 0.0009585
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008416, 0.0008447
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003188, 0.0003199
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012209, 0.0012165
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008543, 0.0008512
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009160, 0.0009127
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006029, 0.0006051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004416
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004416
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003263, 0.0003280
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008282, 0.0008324
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005138, 0.0005164
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009643, 0.0009594
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008424, 0.0008467
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003191, 0.0003207
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012238, 0.0012176
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008564, 0.0008520
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009182, 0.0009135
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006034, 0.0006065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004415
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004415
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003294, 0.0003236
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008360, 0.0008211
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005186, 0.0005094
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009513, 0.0009684
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008503, 0.0008352
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003221, 0.0003164
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012073, 0.0012291
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008448, 0.0008600
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009057, 0.0009221
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006091, 0.0005983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004844
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004844
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003295, 0.0003246
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008363, 0.0008237
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005188, 0.0005110
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009542, 0.0009688
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008506, 0.0008379
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003222, 0.0003174
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012110, 0.0012295
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008474, 0.0008603
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009086, 0.0009224
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006093, 0.0006002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004828
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004828
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003245, 0.0003277
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008235, 0.0008317
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005109, 0.0005160
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009635, 0.0009540
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008376, 0.0008460
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003173, 0.0003204
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012228, 0.0012107
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008556, 0.0008472
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009174, 0.0009083
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006000, 0.0006060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004416
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004416
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003247, 0.0003286
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008239, 0.0008338
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005111, 0.0005173
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009659, 0.0009544
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008380, 0.0008481
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003174, 0.0003212
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012259, 0.0012113
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008578, 0.0008476
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009197, 0.0009088
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006003, 0.0006075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004415
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004415
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003280, 0.0003250
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008323, 0.0008247
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005164, 0.0005116
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009554, 0.0009642
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008466, 0.0008388
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003207, 0.0003177
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012125, 0.0012237
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008484, 0.0008563
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009097, 0.0009181
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006064, 0.0006009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004844
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004844
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003279, 0.0003258
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008321, 0.0008267
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005162, 0.0005129
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009577, 0.0009639
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008463, 0.0008409
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003206, 0.0003185
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012154, 0.0012233
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008505, 0.0008560
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009118, 0.0009178
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006063, 0.0006023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004828
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004828
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003266, 0.0003272
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008288, 0.0008304
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005142, 0.0005152
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009620, 0.0009601
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008430, 0.0008447
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003193, 0.0003199
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012209, 0.0012185
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008543, 0.0008526
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009160, 0.0009142
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006039, 0.0006050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003258, 0.0003273
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008267, 0.0008305
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005129, 0.0005153
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009621, 0.0009577
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008409, 0.0008448
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003185, 0.0003200
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012211, 0.0012154
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008545, 0.0008505
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009161, 0.0009119
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006023, 0.0006051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003294, 0.0003240
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008359, 0.0008222
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005186, 0.0005101
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009525, 0.0009684
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008503, 0.0008363
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003221, 0.0003168
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012088, 0.0012290
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008459, 0.0008600
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009069, 0.0009220
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006091, 0.0005991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004741
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004741
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003285, 0.0003238
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008337, 0.0008217
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005172, 0.0005098
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009519, 0.0009658
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008480, 0.0008358
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003212, 0.0003166
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012081, 0.0012257
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008454, 0.0008577
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009064, 0.0009196
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006074, 0.0005987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004731
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004731
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003254, 0.0003289
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008257, 0.0008345
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005123, 0.0005177
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009667, 0.0009565
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008399, 0.0008488
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003181, 0.0003215
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012269, 0.0012140
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008585, 0.0008495
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009205, 0.0009108
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006016, 0.0006080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003244, 0.0003287
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008231, 0.0008342
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005107, 0.0005175
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009663, 0.0009535
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008372, 0.0008485
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003171, 0.0003214
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012264, 0.0012102
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008582, 0.0008468
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009201, 0.0009079
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005997, 0.0006078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003287, 0.0003256
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008341, 0.0008263
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005175, 0.0005126
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009572, 0.0009662
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008484, 0.0008405
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003213, 0.0003184
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012148, 0.0012263
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008501, 0.0008581
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009114, 0.0009200
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006077, 0.0006021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004741
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004741
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003279, 0.0003253
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008320, 0.0008255
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005162, 0.0005122
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009563, 0.0009638
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008463, 0.0008397
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003205, 0.0003181
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012137, 0.0012232
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008493, 0.0008559
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009106, 0.0009177
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006062, 0.0006015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004731
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004731
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003272, 0.0003254
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008302, 0.0008257
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005151, 0.0005123
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009566, 0.0009618
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008445, 0.0008399
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003199, 0.0003181
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012140, 0.0012206
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008495, 0.0008541
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009108, 0.0009158
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006049, 0.0006016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004416
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004416
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003268, 0.0003258
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008293, 0.0008268
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005145, 0.0005130
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009579, 0.0009607
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008435, 0.0008410
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003195, 0.0003186
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012156, 0.0012193
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008506, 0.0008532
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009120, 0.0009147
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006042, 0.0006024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004415
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004415
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003304, 0.0003222
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008385, 0.0008175
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005202, 0.0005072
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009471, 0.0009714
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008529, 0.0008316
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003231, 0.0003150
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012019, 0.0012328
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008411, 0.0008627
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009018, 0.0009249
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006109, 0.0005957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004844
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004844
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003300, 0.0003224
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008375, 0.0008183
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005196, 0.0005076
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009479, 0.0009703
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008519, 0.0008323
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003227, 0.0003153
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012030, 0.0012314
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008418, 0.0008617
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009026, 0.0009238
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006102, 0.0005962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004828
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004828
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003259, 0.0003270
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008271, 0.0008297
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005131, 0.0005147
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009612, 0.0009582
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008413, 0.0008439
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003187, 0.0003197
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012198, 0.0012160
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008536, 0.0008509
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009152, 0.0009123
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006026, 0.0006045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004416
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004416
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003254, 0.0003270
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008257, 0.0008299
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005123, 0.0005149
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009614, 0.0009566
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008399, 0.0008442
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003181, 0.0003197
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012201, 0.0012140
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008538, 0.0008495
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009154, 0.0009108
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006016, 0.0006047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004415
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004415
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003293, 0.0003241
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008356, 0.0008224
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005184, 0.0005102
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009527, 0.0009681
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008500, 0.0008365
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003220, 0.0003168
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012091, 0.0012286
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008460, 0.0008597
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009071, 0.0009217
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006089, 0.0005992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004844
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004844
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003288, 0.0003239
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008344, 0.0008221
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005177, 0.0005100
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009523, 0.0009666
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008487, 0.0008362
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003215, 0.0003167
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012086, 0.0012267
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008457, 0.0008584
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009068, 0.0009203
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006079, 0.0005990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004828
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004828
time: 0.58 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004152
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004152
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004152
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004152
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004152
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004152
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004152
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004152
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004380
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0004253
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0004253
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004741, upper bound: 0.0004253
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004741, upper bound: 0.0004253
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0004253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0004253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004741, upper bound: 0.0004253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004741, upper bound: 0.0004253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004156
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004156
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004156
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004156
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004402
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004402
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004402
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004402
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004156
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004828, upper bound: 0.0004156
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004156
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004844, upper bound: 0.0004156
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004402
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004415, upper bound: 0.0004402
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004402
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004416, upper bound: 0.0004402
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004251
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004251
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004734, upper bound: 0.0004251
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004734, upper bound: 0.0004251
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004251
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004251
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004734, upper bound: 0.0004251
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004734, upper bound: 0.0004251
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004378, upper bound: 0.0004583
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004734
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004734
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004734
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004734
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004251, upper bound: 0.0004722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004416
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004416
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004415
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004415
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004844
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004844
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004416
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004416
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004415
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004415
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004844
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004844
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004828
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004156, upper bound: 0.0004828
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004741
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004741
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004731
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004731
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004378
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004583, upper bound: 0.0004377
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004741
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004741
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004731
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004253, upper bound: 0.0004731
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004416
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004416
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004415
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004415
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004844
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004844
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004416
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004416
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004415
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004380, upper bound: 0.0004415
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004844
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004844
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.22
Output dim: 7, lower bound: -0.0004152, upper bound: 0.0004828

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003216, 0.0003268
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008161, 0.0008292
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005063, 0.0005144
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009606, 0.0009454
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008301, 0.0008434
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003144, 0.0003195
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012191, 0.0011998
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008531, 0.0008396
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009146, 0.0009002
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005946, 0.0006042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004090
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0004118
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003239, 0.0003264
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008221, 0.0008284
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005100, 0.0005139
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009597, 0.0009523
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008362, 0.0008426
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003167, 0.0003192
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012179, 0.0012086
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008522, 0.0008457
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009137, 0.0009068
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005990, 0.0006036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004090
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0004118
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003217, 0.0003273
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008164, 0.0008305
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005065, 0.0005153
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009621, 0.0009457
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008304, 0.0008448
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003145, 0.0003200
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012211, 0.0012003
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008544, 0.0008399
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009161, 0.0009005
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005948, 0.0006051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004812, upper bound: 0.0004090
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0004118
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003241, 0.0003269
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008224, 0.0008297
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005102, 0.0005147
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009611, 0.0009527
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008365, 0.0008439
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003168, 0.0003196
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012198, 0.0012091
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008535, 0.0008460
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009151, 0.0009071
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005992, 0.0006045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004812, upper bound: 0.0004090
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0004118
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003247, 0.0003234
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008239, 0.0008207
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005112, 0.0005092
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009507, 0.0009545
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008381, 0.0008348
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003174, 0.0003162
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012066, 0.0012113
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008443, 0.0008476
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009052, 0.0009088
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006003, 0.0005980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004336
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004347
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003270, 0.0003230
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008299, 0.0008198
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005149, 0.0005086
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009496, 0.0009614
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008442, 0.0008338
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003197, 0.0003158
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012052, 0.0012201
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008434, 0.0008538
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009042, 0.0009154
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006047, 0.0005973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004336
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004347
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003246, 0.0003239
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008237, 0.0008221
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005110, 0.0005100
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009523, 0.0009542
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008378, 0.0008362
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003174, 0.0003167
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012086, 0.0012110
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008457, 0.0008474
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009068, 0.0009086
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006002, 0.0005990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004336
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004347
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003270, 0.0003236
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008297, 0.0008211
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005147, 0.0005094
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009512, 0.0009612
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008439, 0.0008352
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003197, 0.0003164
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012072, 0.0012198
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008448, 0.0008536
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009057, 0.0009152
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006045, 0.0005983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004336
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004347
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003201, 0.0003281
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008123, 0.0008327
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005039, 0.0005166
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009646, 0.0009410
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008262, 0.0008470
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003129, 0.0003208
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012242, 0.0011942
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008567, 0.0008357
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009185, 0.0008960
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005918, 0.0006067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004090
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0004118
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003224, 0.0003277
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008183, 0.0008316
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005076, 0.0005159
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009633, 0.0009479
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008323, 0.0008458
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003153, 0.0003204
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012226, 0.0012030
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008555, 0.0008418
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009172, 0.0009026
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005962, 0.0006059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004090
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0004118
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003198, 0.0003285
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008115, 0.0008336
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005035, 0.0005172
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009657, 0.0009401
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008255, 0.0008479
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003127, 0.0003212
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012256, 0.0011931
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008576, 0.0008349
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009195, 0.0008951
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005913, 0.0006074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004812, upper bound: 0.0004090
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0004118
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003222, 0.0003281
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008175, 0.0008325
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005072, 0.0005165
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009644, 0.0009471
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008316, 0.0008468
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003150, 0.0003207
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012240, 0.0012019
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008565, 0.0008411
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009183, 0.0009018
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005957, 0.0006066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004812, upper bound: 0.0004090
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0004118
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003235, 0.0003246
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008209, 0.0008238
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005093, 0.0005111
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009543, 0.0009509
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008349, 0.0008379
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003163, 0.0003174
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012111, 0.0012068
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008475, 0.0008445
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009086, 0.0009054
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005981, 0.0006002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004336
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004347
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003258, 0.0003244
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008268, 0.0008233
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005130, 0.0005108
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009538, 0.0009579
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008410, 0.0008375
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003186, 0.0003172
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012105, 0.0012156
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008470, 0.0008506
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009081, 0.0009120
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006024, 0.0005999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004336
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004347
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003230, 0.0003250
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008197, 0.0008246
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005086, 0.0005116
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009553, 0.0009496
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008338, 0.0008388
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003158, 0.0003177
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012124, 0.0012052
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008484, 0.0008433
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009096, 0.0009042
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005973, 0.0006008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004336
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004347
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003254, 0.0003248
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008257, 0.0008243
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005123, 0.0005114
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009549, 0.0009566
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008399, 0.0008384
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003181, 0.0003176
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012118, 0.0012140
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008480, 0.0008495
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009092, 0.0009108
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006016, 0.0006006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004336
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004347
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003230, 0.0003257
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008196, 0.0008265
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005085, 0.0005128
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009575, 0.0009494
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008336, 0.0008407
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003158, 0.0003184
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012152, 0.0012049
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008503, 0.0008431
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009117, 0.0009040
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005971, 0.0006022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004698, upper bound: 0.0004188
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004673, upper bound: 0.0004220
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003253, 0.0003255
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008255, 0.0008260
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005122, 0.0005124
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009569, 0.0009563
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008397, 0.0008402
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003181, 0.0003182
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012144, 0.0012137
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008498, 0.0008493
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009111, 0.0009106
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006015, 0.0006018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004698, upper bound: 0.0004188
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004673, upper bound: 0.0004220
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003233, 0.0003264
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008203, 0.0008282
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005089, 0.0005138
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009594, 0.0009503
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008344, 0.0008424
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003160, 0.0003191
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012176, 0.0012060
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008521, 0.0008439
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009135, 0.0009048
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005977, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004707, upper bound: 0.0004188
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004674, upper bound: 0.0004220
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003256, 0.0003263
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008263, 0.0008281
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005126, 0.0005137
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009593, 0.0009572
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008405, 0.0008423
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003184, 0.0003190
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012175, 0.0012148
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008519, 0.0008501
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009134, 0.0009114
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006021, 0.0006034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004707, upper bound: 0.0004188
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004674, upper bound: 0.0004220
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003264, 0.0003224
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008282, 0.0008180
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005138, 0.0005075
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009476, 0.0009594
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008424, 0.0008321
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003191, 0.0003152
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012027, 0.0012176
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008416, 0.0008520
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009023, 0.0009135
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006034, 0.0005960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004344, upper bound: 0.0004499
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003287, 0.0003220
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008342, 0.0008171
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005175, 0.0005070
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009466, 0.0009663
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008485, 0.0008312
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003214, 0.0003148
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012014, 0.0012264
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008407, 0.0008582
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009013, 0.0009201
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006078, 0.0005954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004344, upper bound: 0.0004499
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003265, 0.0003232
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008285, 0.0008201
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005140, 0.0005088
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009500, 0.0009598
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008427, 0.0008341
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003192, 0.0003160
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012057, 0.0012181
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008437, 0.0008524
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009046, 0.0009139
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006037, 0.0005975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004345, upper bound: 0.0004499
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003289, 0.0003230
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008345, 0.0008197
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005177, 0.0005086
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009496, 0.0009667
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008488, 0.0008338
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003215, 0.0003158
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012052, 0.0012269
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008433, 0.0008585
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009042, 0.0009205
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006080, 0.0005973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004345, upper bound: 0.0004499
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003215, 0.0003265
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008157, 0.0008287
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005061, 0.0005141
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009600, 0.0009450
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008297, 0.0008429
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003143, 0.0003193
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012183, 0.0011993
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008525, 0.0008392
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009140, 0.0008998
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005944, 0.0006038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004698, upper bound: 0.0004188
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004673, upper bound: 0.0004220
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003238, 0.0003262
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008217, 0.0008277
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005098, 0.0005135
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009589, 0.0009519
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008358, 0.0008419
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003166, 0.0003189
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012169, 0.0012081
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008516, 0.0008454
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009130, 0.0009064
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005987, 0.0006031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004698, upper bound: 0.0004188
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004673, upper bound: 0.0004220
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003216, 0.0003273
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008162, 0.0008304
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005064, 0.0005152
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009620, 0.0009456
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008302, 0.0008447
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003145, 0.0003200
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012209, 0.0012000
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008544, 0.0008397
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009160, 0.0009003
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005947, 0.0006051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004707, upper bound: 0.0004188
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004674, upper bound: 0.0004220
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003240, 0.0003270
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008222, 0.0008299
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005101, 0.0005149
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009614, 0.0009525
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008363, 0.0008442
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003168, 0.0003197
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012202, 0.0012088
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008538, 0.0008459
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009154, 0.0009069
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005991, 0.0006047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004707, upper bound: 0.0004188
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004674, upper bound: 0.0004220
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003249, 0.0003236
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008246, 0.0008212
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005116, 0.0005095
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009513, 0.0009552
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008387, 0.0008353
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003177, 0.0003164
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012073, 0.0012123
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008448, 0.0008483
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009058, 0.0009095
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006008, 0.0005983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004344, upper bound: 0.0004499
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003273, 0.0003234
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008305, 0.0008207
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005153, 0.0005092
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009507, 0.0009621
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008448, 0.0008348
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003200, 0.0003162
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012066, 0.0012211
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008443, 0.0008545
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009052, 0.0009161
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006051, 0.0005980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004344, upper bound: 0.0004499
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003249, 0.0003244
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008244, 0.0008233
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005115, 0.0005108
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009538, 0.0009550
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008386, 0.0008375
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003176, 0.0003172
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012105, 0.0012121
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008470, 0.0008482
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009081, 0.0009094
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006007, 0.0005999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004345, upper bound: 0.0004499
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003272, 0.0003242
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008304, 0.0008228
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005152, 0.0005105
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009532, 0.0009620
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008447, 0.0008369
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003199, 0.0003170
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012097, 0.0012209
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008465, 0.0008543
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009076, 0.0009160
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006050, 0.0005995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004345, upper bound: 0.0004499
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003234, 0.0003258
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008207, 0.0008268
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005092, 0.0005130
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009578, 0.0009507
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008348, 0.0008410
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003162, 0.0003186
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012156, 0.0012066
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008506, 0.0008443
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009120, 0.0009052
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005980, 0.0006024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004090
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004762, upper bound: 0.0004122
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003258, 0.0003255
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008267, 0.0008261
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005129, 0.0005125
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009570, 0.0009577
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008409, 0.0008403
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003185, 0.0003183
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012145, 0.0012154
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008499, 0.0008505
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009112, 0.0009118
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006023, 0.0006019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004090
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004762, upper bound: 0.0004122
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003226, 0.0003260
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008187, 0.0008273
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005079, 0.0005132
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009584, 0.0009484
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008328, 0.0008415
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003154, 0.0003187
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012163, 0.0012037
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008511, 0.0008423
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009125, 0.0009031
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005965, 0.0006028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004812, upper bound: 0.0004090
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004764, upper bound: 0.0004122
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003250, 0.0003256
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008247, 0.0008263
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005116, 0.0005127
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009573, 0.0009554
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008388, 0.0008405
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003177, 0.0003184
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012149, 0.0012125
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008501, 0.0008484
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009115, 0.0009097
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006009, 0.0006021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004812, upper bound: 0.0004090
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004764, upper bound: 0.0004122
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003262, 0.0003225
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008278, 0.0008184
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005136, 0.0005077
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009481, 0.0009590
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008420, 0.0008324
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003189, 0.0003153
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012032, 0.0012171
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008419, 0.0008516
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009027, 0.0009131
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006032, 0.0005963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004351
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004369
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003286, 0.0003223
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008338, 0.0008179
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005173, 0.0005074
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009475, 0.0009659
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008481, 0.0008319
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003212, 0.0003151
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012025, 0.0012259
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008414, 0.0008578
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009021, 0.0009197
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006075, 0.0005959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004351
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004369
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003254, 0.0003225
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008257, 0.0008185
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005123, 0.0005078
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009482, 0.0009565
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008399, 0.0008325
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003181, 0.0003153
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012033, 0.0012140
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008420, 0.0008495
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009028, 0.0009108
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006016, 0.0005963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004351
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004369
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003277, 0.0003222
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008317, 0.0008175
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005160, 0.0005072
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009471, 0.0009635
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008460, 0.0008315
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003204, 0.0003150
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012019, 0.0012228
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008411, 0.0008556
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009017, 0.0009174
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006060, 0.0005957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004351
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004369
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003222, 0.0003273
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008177, 0.0008307
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005073, 0.0005154
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009623, 0.0009473
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008318, 0.0008449
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003150, 0.0003200
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012213, 0.0012022
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008546, 0.0008413
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009163, 0.0009020
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005958, 0.0006052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004090
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004762, upper bound: 0.0004122
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003246, 0.0003272
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008237, 0.0008303
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005110, 0.0005151
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009618, 0.0009542
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008379, 0.0008445
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003174, 0.0003199
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012207, 0.0012110
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008542, 0.0008474
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009158, 0.0009086
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006002, 0.0006049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004090
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004762, upper bound: 0.0004122
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003212, 0.0003274
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008152, 0.0008309
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005057, 0.0005155
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009626, 0.0009443
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008291, 0.0008452
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003141, 0.0003201
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012216, 0.0011985
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008548, 0.0008386
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009165, 0.0008991
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005939, 0.0006054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004812, upper bound: 0.0004090
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004764, upper bound: 0.0004122
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003236, 0.0003271
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008211, 0.0008300
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005094, 0.0005149
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009615, 0.0009513
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008352, 0.0008442
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003164, 0.0003198
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012203, 0.0012073
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008539, 0.0008448
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009155, 0.0009057
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005983, 0.0006047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004812, upper bound: 0.0004090
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004764, upper bound: 0.0004122
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003257, 0.0003241
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008264, 0.0008224
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005127, 0.0005102
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009527, 0.0009574
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008406, 0.0008365
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003184, 0.0003168
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012091, 0.0012150
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008461, 0.0008502
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009071, 0.0009116
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006021, 0.0005992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004351
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004369
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003280, 0.0003240
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008324, 0.0008222
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005164, 0.0005101
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009524, 0.0009643
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008467, 0.0008363
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003207, 0.0003168
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012088, 0.0012238
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008458, 0.0008564
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009069, 0.0009182
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006065, 0.0005990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004351
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004369
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003249, 0.0003241
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008244, 0.0008223
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005115, 0.0005102
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009526, 0.0009551
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008386, 0.0008364
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003176, 0.0003168
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012090, 0.0012121
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008460, 0.0008482
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009071, 0.0009094
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006007, 0.0005992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004351
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004369
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003272, 0.0003237
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008304, 0.0008214
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005152, 0.0005096
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009516, 0.0009620
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008447, 0.0008355
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003199, 0.0003165
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012077, 0.0012209
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008451, 0.0008543
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009061, 0.0009160
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006051, 0.0005985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004381, upper bound: 0.0004351
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004369, upper bound: 0.0004369
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003240, 0.0003242
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008223, 0.0008226
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005101, 0.0005104
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009530, 0.0009526
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008364, 0.0008368
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003168, 0.0003169
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012095, 0.0012089
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008463, 0.0008459
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009074, 0.0009070
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005991, 0.0005994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004688, upper bound: 0.0004188
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004672, upper bound: 0.0004218
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003264, 0.0003237
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008283, 0.0008214
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005139, 0.0005096
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009516, 0.0009595
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008425, 0.0008355
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003191, 0.0003165
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012077, 0.0012177
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008451, 0.0008521
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009061, 0.0009136
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006035, 0.0005985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004688, upper bound: 0.0004188
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004672, upper bound: 0.0004218
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003237, 0.0003246
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008215, 0.0008236
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005097, 0.0005110
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009541, 0.0009517
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008357, 0.0008378
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003165, 0.0003173
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012109, 0.0012079
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008473, 0.0008452
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009085, 0.0009062
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005986, 0.0006001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004699, upper bound: 0.0004188
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004673, upper bound: 0.0004218
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003261, 0.0003242
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008275, 0.0008226
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005134, 0.0005103
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009530, 0.0009587
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008417, 0.0008367
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003188, 0.0003169
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012094, 0.0012167
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008463, 0.0008514
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009074, 0.0009128
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006030, 0.0005994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004699, upper bound: 0.0004188
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004673, upper bound: 0.0004218
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003273, 0.0003207
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008306, 0.0008138
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005153, 0.0005049
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009427, 0.0009622
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008449, 0.0008278
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003200, 0.0003135
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0011964, 0.0012212
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008372, 0.0008545
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0008976, 0.0009162
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006052, 0.0005929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004344, upper bound: 0.0004499
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003297, 0.0003205
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008366, 0.0008133
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005190, 0.0005046
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009421, 0.0009691
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008509, 0.0008272
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003223, 0.0003133
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0011957, 0.0012300
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008367, 0.0008607
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0008971, 0.0009228
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006095, 0.0005926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004344, upper bound: 0.0004499
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003270, 0.0003211
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008298, 0.0008149
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005148, 0.0005056
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009440, 0.0009613
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008440, 0.0008289
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003197, 0.0003140
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0011980, 0.0012200
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008383, 0.0008537
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0008988, 0.0009153
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006046, 0.0005937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004345, upper bound: 0.0004499
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003294, 0.0003208
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008358, 0.0008141
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005185, 0.0005051
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009431, 0.0009682
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008501, 0.0008281
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003220, 0.0003137
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0011969, 0.0012288
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008376, 0.0008598
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0008980, 0.0009219
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006090, 0.0005932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004345, upper bound: 0.0004499
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003228, 0.0003257
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008192, 0.0008265
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005082, 0.0005128
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009575, 0.0009490
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008333, 0.0008407
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003156, 0.0003184
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012151, 0.0012044
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008503, 0.0008428
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009117, 0.0009036
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005969, 0.0006022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004688, upper bound: 0.0004188
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004672, upper bound: 0.0004218
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003252, 0.0003253
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008252, 0.0008255
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005120, 0.0005122
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009563, 0.0009560
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008394, 0.0008397
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003179, 0.0003181
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012137, 0.0012132
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008493, 0.0008490
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009106, 0.0009102
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006013, 0.0006015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004688, upper bound: 0.0004188
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004672, upper bound: 0.0004218
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003224, 0.0003259
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008181, 0.0008270
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005075, 0.0005131
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009580, 0.0009477
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008321, 0.0008412
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003152, 0.0003186
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012159, 0.0012028
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008508, 0.0008416
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009122, 0.0009024
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0005961, 0.0006026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004699, upper bound: 0.0004188
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004673, upper bound: 0.0004218
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003247, 0.0003255
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008241, 0.0008259
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005113, 0.0005124
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009568, 0.0009546
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008382, 0.0008401
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003175, 0.0003182
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012142, 0.0012116
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008497, 0.0008478
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009110, 0.0009090
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006004, 0.0006018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004699, upper bound: 0.0004188
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004673, upper bound: 0.0004218
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003263, 0.0003226
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008281, 0.0008187
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005138, 0.0005079
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009485, 0.0009593
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008423, 0.0008328
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003191, 0.0003154
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012037, 0.0012175
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008423, 0.0008520
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009031, 0.0009134
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006034, 0.0005965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004344, upper bound: 0.0004499
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003287, 0.0003224
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008341, 0.0008182
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005175, 0.0005076
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009479, 0.0009663
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008484, 0.0008323
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003214, 0.0003152
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012030, 0.0012263
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008418, 0.0008581
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009025, 0.0009200
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006077, 0.0005962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004344, upper bound: 0.0004499
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004331, upper bound: 0.0004550
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0011586, -0.0005350, -0.0011586, -0.0005350, -0.0003258, 0.0003227
1: -0.0072507, -0.0056682, -0.0072507, -0.0056682, -0.0008269, 0.0008189
2: 0.0305317, 0.0315134, 0.0305317, 0.0315134, -0.0005130, 0.0005080
3: 0.0007190, 0.0025521, 0.0007190, 0.0025521, -0.0009487, 0.0009579
4: -0.0062682, -0.0046586, -0.0062682, -0.0046586, -0.0008411, 0.0008330
5: 0.0113640, 0.0119736, 0.0113640, 0.0119736, -0.0003186, 0.0003155
6: 0.0013006, 0.0036271, 0.0013006, 0.0036271, -0.0012040, 0.0012157
7: 0.9789694, 0.9805974, 0.9789694, 0.9805974, -0.0008425, 0.0008507
8: -0.0091124, -0.0073669, -0.0091124, -0.0073669, -0.0009033, 0.0009121
9: -0.0001333, 0.0010197, -0.0001333, 0.0010197, -0.0006025, 0.0005967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.18 + 597.38 = 600.55 seconds
