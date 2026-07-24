## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00206416


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0028122, 0.0028122)
1: (0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0004063, 0.0004063)
2: (0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0015548, 0.0015548)
3: (-0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0016080, 0.0016080)
4: (-0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0017408, 0.0017408)
5: (0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0016474, 0.0016474)
6: (-0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0065363, 0.0065363)
7: (0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0089019, 0.0089019)
8: (0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0062707, 0.0062707)
9: (-0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0056921, 0.0056921)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.68 + 1.60 = 3.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0030595, upper bound: 0.0030595

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0029943, upper bound: 0.0029706
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0029706, upper bound: 0.0029943
time: 0.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.34 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 8, lower bound: -0.0029943, upper bound: 0.0029706
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 8, lower bound: -0.0029706, upper bound: 0.0029943

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027667, 0.0027608
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003997, 0.0003989
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0015264, 0.0015296
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015786, 0.0015820
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0017126, 0.0017090
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0016173, 0.0016207
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0064168, 0.0064305
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0087578, 0.0087391
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0061692, 0.0061560
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0055880, 0.0056000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027248, upper bound: 0.0028453
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0028711, upper bound: 0.0027089
time: 0.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027608, 0.0027667
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003989, 0.0003997
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0015296, 0.0015264
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015820, 0.0015786
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0017090, 0.0017126
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0016207, 0.0016173
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0064305, 0.0064168
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0087391, 0.0087578
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0061560, 0.0061692
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0056000, 0.0055880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027722, upper bound: 0.0027882
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027727, upper bound: 0.0027882
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.72 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.72
Output dim: 8, lower bound: -0.0027248, upper bound: 0.0028453
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.72
Output dim: 8, lower bound: -0.0028711, upper bound: 0.0027089
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.72
Output dim: 8, lower bound: -0.0027722, upper bound: 0.0027882
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.72
Output dim: 8, lower bound: -0.0027727, upper bound: 0.0027882

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027035, 0.0027176
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003906, 0.0003926
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0015025, 0.0014947
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015539, 0.0015459
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016735, 0.0016822
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015919, 0.0015837
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0063164, 0.0062836
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0085578, 0.0086024
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060283, 0.0060597
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0055006, 0.0054721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024449, upper bound: 0.0024969
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024449, upper bound: 0.0024969
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027255, 0.0026976
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003938, 0.0003897
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014914, 0.0015069
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015425, 0.0015585
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016871, 0.0016699
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015802, 0.0015966
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0062699, 0.0063348
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0086274, 0.0085391
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060773, 0.0060151
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0054601, 0.0055166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0028701, upper bound: 0.0027036
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0028597, upper bound: 0.0027075
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027431, 0.0027615
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003963, 0.0003990
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0015268, 0.0015166
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015790, 0.0015685
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016980, 0.0017094
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0016177, 0.0016069
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0064185, 0.0063756
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0086831, 0.0087414
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0061165, 0.0061576
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0055895, 0.0055522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027721, upper bound: 0.0027786
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027674, upper bound: 0.0027865
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027608, 0.0027490
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003989, 0.0003971
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0015198, 0.0015264
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015719, 0.0015786
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0017090, 0.0017016
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0016103, 0.0016173
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0063893, 0.0064168
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0087391, 0.0087017
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0061560, 0.0061297
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0055641, 0.0055880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024723, upper bound: 0.0026220
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026076, upper bound: 0.0024955
time: 0.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 8, lower bound: -0.0024449, upper bound: 0.0024969
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 8, lower bound: -0.0024449, upper bound: 0.0024969
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 8, lower bound: -0.0028701, upper bound: 0.0027036
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 8, lower bound: -0.0028597, upper bound: 0.0027075
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 8, lower bound: -0.0027721, upper bound: 0.0027786
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 8, lower bound: -0.0027674, upper bound: 0.0027865
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 8, lower bound: -0.0024723, upper bound: 0.0026220
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 8, lower bound: -0.0026076, upper bound: 0.0024955

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0026984, 0.0027060
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003898, 0.0003909
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014961, 0.0014919
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015473, 0.0015430
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016704, 0.0016751
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015852, 0.0015807
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0062896, 0.0062719
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0085418, 0.0085658
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060170, 0.0060340
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0054772, 0.0054618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023337, upper bound: 0.0023706
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023337, upper bound: 0.0023706
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027035, 0.0027125
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003906, 0.0003919
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014997, 0.0014947
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015510, 0.0015459
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016735, 0.0016791
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015890, 0.0015837
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0063046, 0.0062836
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0085578, 0.0085864
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060283, 0.0060484
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0054904, 0.0054721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021232, upper bound: 0.0021179
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021232, upper bound: 0.0021179
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027107, 0.0026780
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003916, 0.0003869
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014806, 0.0014987
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015313, 0.0015500
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016780, 0.0016577
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015688, 0.0015879
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0062245, 0.0063004
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0085806, 0.0084772
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060444, 0.0059715
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0054206, 0.0054867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0028075, upper bound: 0.0026991
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0028646, upper bound: 0.0026488
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027059, 0.0026826
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003909, 0.0003876
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014831, 0.0014960
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015339, 0.0015473
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016750, 0.0016605
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015714, 0.0015851
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0062350, 0.0062893
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0085655, 0.0084915
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060337, 0.0059816
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0054297, 0.0054770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026926, upper bound: 0.0024919
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026491, upper bound: 0.0025475
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027290, 0.0027429
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003943, 0.0003963
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0015165, 0.0015088
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015684, 0.0015605
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016893, 0.0016979
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0016068, 0.0015987
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0063754, 0.0063430
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0086387, 0.0086827
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060853, 0.0061163
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0055519, 0.0055238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021391, upper bound: 0.0021461
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021391, upper bound: 0.0021461
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027245, 0.0027478
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003936, 0.0003970
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0015192, 0.0015063
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015712, 0.0015579
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016865, 0.0017009
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0016097, 0.0015960
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0063866, 0.0063325
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0086244, 0.0086980
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060752, 0.0061271
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0055618, 0.0055147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027163, upper bound: 0.0027836
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027646, upper bound: 0.0027325
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023215, 0.0023503
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003354, 0.0003396
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012994, 0.0012835
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013439, 0.0013274
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014370, 0.0014549
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013768, 0.0013599
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0054628, 0.0053958
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0073486, 0.0074399
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0051765, 0.0052408
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0047572, 0.0046989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020800, upper bound: 0.0021710
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020800, upper bound: 0.0021710
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023640, 0.0023098
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003415, 0.0003337
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012770, 0.0013070
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013208, 0.0013518
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014634, 0.0014298
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013531, 0.0013849
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053686, 0.0054947
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0074833, 0.0073116
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0052714, 0.0051504
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046752, 0.0047850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020896
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020896
time: 0.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0023337, upper bound: 0.0023706
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0023337, upper bound: 0.0023706
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0021232, upper bound: 0.0021179
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0021232, upper bound: 0.0021179
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0028075, upper bound: 0.0026991
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0028646, upper bound: 0.0026488
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0026926, upper bound: 0.0024919
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0026491, upper bound: 0.0025475
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0021391, upper bound: 0.0021461
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0021391, upper bound: 0.0021461
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0027163, upper bound: 0.0027836
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0027646, upper bound: 0.0027325
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0020800, upper bound: 0.0021710
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0020800, upper bound: 0.0021710
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020896
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020896

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0026802, 0.0026855
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003872, 0.0003880
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014847, 0.0014818
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015356, 0.0015326
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016591, 0.0016623
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015731, 0.0015700
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0062417, 0.0062295
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0084840, 0.0085007
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0059763, 0.0059881
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0054356, 0.0054249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020866, upper bound: 0.0022182
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021694, upper bound: 0.0021179
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0026769, 0.0026878
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003867, 0.0003883
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014860, 0.0014800
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015369, 0.0015307
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016570, 0.0016638
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015745, 0.0015681
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0062472, 0.0062218
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0084735, 0.0085081
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0059689, 0.0059933
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0054403, 0.0054182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020866, upper bound: 0.0022182
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021694, upper bound: 0.0021179
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0026526, 0.0026626
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003832, 0.0003847
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014721, 0.0014666
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015225, 0.0015168
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016420, 0.0016482
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015597, 0.0015539
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0061885, 0.0061654
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0083967, 0.0084282
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0059148, 0.0059370
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0053892, 0.0053691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020557, upper bound: 0.0020438
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020557, upper bound: 0.0020438
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027035, 0.0026621
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003906, 0.0003846
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014718, 0.0014947
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015222, 0.0015459
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016735, 0.0016479
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015595, 0.0015837
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0061876, 0.0062836
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0085578, 0.0084269
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060283, 0.0059361
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0053884, 0.0054721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018913, upper bound: 0.0019558
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019607, upper bound: 0.0018844
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025265, 0.0024919
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003650, 0.0003600
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013777, 0.0013968
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014249, 0.0014447
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015639, 0.0015425
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014597, 0.0014800
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0057917, 0.0058722
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079974, 0.0078879
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0056336, 0.0055564
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0050437, 0.0051138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024047, upper bound: 0.0023916
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024047, upper bound: 0.0023916
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025242, 0.0024938
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003647, 0.0003603
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013788, 0.0013956
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014260, 0.0014434
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015625, 0.0015437
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014609, 0.0014787
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0057963, 0.0058669
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079902, 0.0078940
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0056285, 0.0055607
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0050476, 0.0051092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027945, upper bound: 0.0025791
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0028005, upper bound: 0.0025791
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0026305, 0.0025885
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003800, 0.0003740
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014311, 0.0014543
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014801, 0.0015041
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016283, 0.0016023
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015163, 0.0015409
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0060164, 0.0061140
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0083267, 0.0081938
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0058655, 0.0057719
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0052393, 0.0053243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026182, upper bound: 0.0024253
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026190, upper bound: 0.0024230
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0026119, 0.0026097
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003773, 0.0003770
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014428, 0.0014440
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014922, 0.0014935
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016168, 0.0016154
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015287, 0.0015300
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0060655, 0.0060707
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0082678, 0.0082607
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0058240, 0.0058190
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0052821, 0.0052867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023262, upper bound: 0.0023479
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024717, upper bound: 0.0022563
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027100, 0.0027194
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003915, 0.0003929
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0015035, 0.0014983
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015550, 0.0015496
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016775, 0.0016833
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015930, 0.0015875
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0063205, 0.0062988
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0085784, 0.0086080
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060428, 0.0060637
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0055042, 0.0054852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019066, upper bound: 0.0019788
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019726, upper bound: 0.0019111
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0027068, 0.0027239
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003911, 0.0003935
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0015060, 0.0014965
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0015575, 0.0015478
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016756, 0.0016861
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015956, 0.0015856
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0063311, 0.0062914
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0085683, 0.0086224
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0060357, 0.0060738
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0055134, 0.0054788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021081, upper bound: 0.0021400
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021330, upper bound: 0.0021104
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025076, 0.0025304
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003623, 0.0003656
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013990, 0.0013864
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014469, 0.0014339
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015522, 0.0015664
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014823, 0.0014689
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0058814, 0.0058283
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079377, 0.0080100
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0055915, 0.0056424
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0051218, 0.0050756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025514, upper bound: 0.0026073
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025514, upper bound: 0.0026073
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025030, 0.0025309
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003616, 0.0003656
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013992, 0.0013838
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014472, 0.0014312
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015494, 0.0015666
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014826, 0.0014663
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0058824, 0.0058177
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079232, 0.0080114
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0055812, 0.0056434
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0051227, 0.0050663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024646, upper bound: 0.0025710
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025950, upper bound: 0.0024279
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023035, 0.0023310
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003328, 0.0003368
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012887, 0.0012735
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013329, 0.0013172
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014259, 0.0014429
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013655, 0.0013494
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0054179, 0.0053539
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0072916, 0.0073787
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0051363, 0.0051977
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0047181, 0.0046624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020207, upper bound: 0.0021650
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0021215
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023002, 0.0023318
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003323, 0.0003369
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012892, 0.0012717
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013334, 0.0013153
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014239, 0.0014434
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013660, 0.0013475
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0054198, 0.0053464
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0072813, 0.0073813
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0051291, 0.0051996
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0047198, 0.0046559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020207, upper bound: 0.0021650
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0021215
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023460, 0.0022880
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003389, 0.0003305
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012650, 0.0012971
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013083, 0.0013415
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014522, 0.0014163
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013403, 0.0013743
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053179, 0.0054528
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0074263, 0.0072426
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0052312, 0.0051018
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046311, 0.0047486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021122, upper bound: 0.0020835
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021546, upper bound: 0.0020322
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023447, 0.0022913
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003387, 0.0003310
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012668, 0.0012963
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013102, 0.0013407
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014514, 0.0014184
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013422, 0.0013735
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053256, 0.0054497
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0074220, 0.0072531
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0052282, 0.0051092
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046378, 0.0047459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021289, upper bound: 0.0020868
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021578, upper bound: 0.0020088
time: 0.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0020866, upper bound: 0.0022182
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0021694, upper bound: 0.0021179
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0020866, upper bound: 0.0022182
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0021694, upper bound: 0.0021179
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0020557, upper bound: 0.0020438
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0020557, upper bound: 0.0020438
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0018913, upper bound: 0.0019558
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0019607, upper bound: 0.0018844
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0024047, upper bound: 0.0023916
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0024047, upper bound: 0.0023916
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0027945, upper bound: 0.0025791
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0028005, upper bound: 0.0025791
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0026182, upper bound: 0.0024253
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0026190, upper bound: 0.0024230
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0023262, upper bound: 0.0023479
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0024717, upper bound: 0.0022563
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0019066, upper bound: 0.0019788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0019726, upper bound: 0.0019111
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0021081, upper bound: 0.0021400
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0021330, upper bound: 0.0021104
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0025514, upper bound: 0.0026073
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0025514, upper bound: 0.0026073
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0024646, upper bound: 0.0025710
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0025950, upper bound: 0.0024279
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0020207, upper bound: 0.0021650
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0021215
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0020207, upper bound: 0.0021650
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0020741, upper bound: 0.0021215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0021122, upper bound: 0.0020835
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0021546, upper bound: 0.0020322
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0021289, upper bound: 0.0020868
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.77
Output dim: 8, lower bound: -0.0021578, upper bound: 0.0020088

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022415, 0.0023098
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003238, 0.0003337
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012770, 0.0012392
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013208, 0.0012817
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0013875, 0.0014298
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013531, 0.0013130
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053686, 0.0052098
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0070953, 0.0073116
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0049980, 0.0051504
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046752, 0.0045369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016781, upper bound: 0.0017160
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016781, upper bound: 0.0017160
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022820, 0.0022467
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003297, 0.0003246
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012422, 0.0012616
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012847, 0.0013049
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014126, 0.0013908
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013161, 0.0013368
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0052220, 0.0053040
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0072235, 0.0071119
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050884, 0.0050098
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0045476, 0.0046189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021061, upper bound: 0.0020520
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021061, upper bound: 0.0020520
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022381, 0.0023080
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003233, 0.0003334
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012760, 0.0012374
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013197, 0.0012798
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0013855, 0.0014287
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013520, 0.0013111
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053643, 0.0052021
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0070848, 0.0073058
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0049907, 0.0051463
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046715, 0.0045302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020369, upper bound: 0.0022133
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020816, upper bound: 0.0021700
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022811, 0.0022491
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003296, 0.0003249
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012434, 0.0012612
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012860, 0.0013044
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014121, 0.0013922
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013175, 0.0013363
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0052274, 0.0053020
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0072209, 0.0071193
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050865, 0.0050150
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0045523, 0.0046172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017346, upper bound: 0.0016491
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017346, upper bound: 0.0016491
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025212, 0.0024803
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003642, 0.0003583
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013713, 0.0013939
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014183, 0.0014417
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015607, 0.0015354
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014530, 0.0014769
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0057649, 0.0058600
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079808, 0.0078513
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0056219, 0.0055306
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0050204, 0.0051031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021493, upper bound: 0.0022201
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022375, upper bound: 0.0021392
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025265, 0.0024866
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003650, 0.0003592
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013748, 0.0013968
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014219, 0.0014447
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015639, 0.0015392
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014566, 0.0014800
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0057795, 0.0058722
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079974, 0.0078712
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0056336, 0.0055447
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0050331, 0.0051138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021493, upper bound: 0.0022201
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022375, upper bound: 0.0021392
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025047, 0.0024710
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003619, 0.0003570
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013662, 0.0013848
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014129, 0.0014322
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015504, 0.0015296
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014475, 0.0014672
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0057433, 0.0058215
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079284, 0.0078219
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0055849, 0.0055099
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0050015, 0.0050696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027382, upper bound: 0.0025765
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027921, upper bound: 0.0025744
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025007, 0.0024743
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003613, 0.0003575
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013680, 0.0013826
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014148, 0.0014299
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015480, 0.0015316
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014494, 0.0014649
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0057509, 0.0058122
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079158, 0.0078322
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0055760, 0.0055171
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0050081, 0.0050616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024746, upper bound: 0.0023872
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026344, upper bound: 0.0022876
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0026118, 0.0025667
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003773, 0.0003708
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014190, 0.0014440
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014676, 0.0014935
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016168, 0.0015888
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015036, 0.0015300
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0059657, 0.0060706
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0082677, 0.0081247
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0058239, 0.0057232
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0051952, 0.0052866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022569, upper bound: 0.0021564
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022569, upper bound: 0.0021564
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0026100, 0.0025699
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003771, 0.0003713
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014208, 0.0014430
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014695, 0.0014924
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016156, 0.0015908
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0015054, 0.0015289
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0059731, 0.0060664
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0082619, 0.0081348
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0058199, 0.0057303
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0052016, 0.0052829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025436, upper bound: 0.0024177
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026134, upper bound: 0.0023757
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0021820, 0.0022320
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003152, 0.0003225
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012340, 0.0012064
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012763, 0.0012477
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0013507, 0.0013816
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013075, 0.0012782
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0051877, 0.0050715
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0069069, 0.0070652
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0048654, 0.0049769
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0045177, 0.0044165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022645, upper bound: 0.0023449
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023234, upper bound: 0.0023362
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022338, 0.0021797
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003227, 0.0003149
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012051, 0.0012350
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012464, 0.0012773
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0013827, 0.0013493
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012769, 0.0013085
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0050663, 0.0051919
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0070710, 0.0068999
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0049809, 0.0048604
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0044120, 0.0045214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024177, upper bound: 0.0022534
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024690, upper bound: 0.0022161
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025222, 0.0025422
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003644, 0.0003673
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014055, 0.0013945
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014537, 0.0014422
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015613, 0.0015737
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014892, 0.0014775
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0059088, 0.0058623
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079840, 0.0080473
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0056241, 0.0056687
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0051457, 0.0051052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018630, upper bound: 0.0019727
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019422, upper bound: 0.0019049
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025207, 0.0025393
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003642, 0.0003669
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0014039, 0.0013936
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014520, 0.0014413
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015603, 0.0015719
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014875, 0.0014766
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0059021, 0.0058587
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079791, 0.0080381
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0056206, 0.0056622
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0051398, 0.0051020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019004, upper bound: 0.0019430
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019664, upper bound: 0.0018699
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0024577, 0.0024817
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003551, 0.0003585
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013720, 0.0013588
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014190, 0.0014053
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015214, 0.0015362
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014537, 0.0014397
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0057681, 0.0057124
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0077797, 0.0078556
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0054802, 0.0055336
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0050231, 0.0049746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020735, upper bound: 0.0021117
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020752, upper bound: 0.0021042
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025076, 0.0024805
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003623, 0.0003584
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013714, 0.0013864
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014184, 0.0014339
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015522, 0.0015355
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014531, 0.0014689
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0057655, 0.0058283
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0079377, 0.0078521
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0055915, 0.0055312
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0050208, 0.0050756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025228, upper bound: 0.0026016
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025455, upper bound: 0.0025525
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019998, 0.0020749
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002889, 0.0002998
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011472, 0.0011056
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011865, 0.0011435
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012379, 0.0012844
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012155, 0.0011715
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0048227, 0.0046480
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0063302, 0.0065681
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0044591, 0.0046267
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041998, 0.0040477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023166, upper bound: 0.0024192
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023166, upper bound: 0.0024192
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020393, 0.0020276
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002946, 0.0002929
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011210, 0.0011275
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011594, 0.0011661
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012623, 0.0012551
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011878, 0.0011946
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047128, 0.0047398
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0064552, 0.0064184
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045472, 0.0045213
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041041, 0.0041276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025334, upper bound: 0.0024220
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025893, upper bound: 0.0024053
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020468, 0.0020754
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002957, 0.0002998
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011474, 0.0011316
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011867, 0.0011704
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012670, 0.0012847
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012158, 0.0011990
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0048239, 0.0047574
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0064791, 0.0065697
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045640, 0.0046278
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0042008, 0.0041429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018630, upper bound: 0.0019727
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018630, upper bound: 0.0019727
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020449, 0.0020740
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002954, 0.0002996
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011467, 0.0011306
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011860, 0.0011693
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012658, 0.0012839
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012150, 0.0011979
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0048206, 0.0047528
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0064730, 0.0065653
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045597, 0.0046247
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041980, 0.0041390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019004, upper bound: 0.0019430
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019004, upper bound: 0.0019430
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020436, 0.0020757
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002952, 0.0002999
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011476, 0.0011298
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011869, 0.0011685
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012650, 0.0012849
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012159, 0.0011971
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0048244, 0.0047498
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0064688, 0.0065705
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045568, 0.0046284
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0042013, 0.0041363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014716, upper bound: 0.0015646
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014716, upper bound: 0.0015646
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020417, 0.0020749
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002950, 0.0002998
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011471, 0.0011288
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011864, 0.0011675
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012638, 0.0012844
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012155, 0.0011960
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0048226, 0.0047454
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0064629, 0.0065680
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045526, 0.0046266
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041997, 0.0041325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017159, upper bound: 0.0017527
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017244, upper bound: 0.0017437
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020894, 0.0020293
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003019, 0.0002932
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011219, 0.0011552
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011604, 0.0011947
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012934, 0.0012562
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011888, 0.0012239
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047167, 0.0048563
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0066138, 0.0064237
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0046589, 0.0045250
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041075, 0.0042291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017341, upper bound: 0.0017365
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017422, upper bound: 0.0017299
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020899, 0.0020310
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003019, 0.0002934
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011229, 0.0011554
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011614, 0.0011950
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012937, 0.0012573
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011898, 0.0012242
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047207, 0.0048574
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0066154, 0.0064292
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0046600, 0.0045289
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041110, 0.0042301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015644, upper bound: 0.0014748
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015644, upper bound: 0.0014748
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020497, 0.0019938
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002961, 0.0002880
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011023, 0.0011332
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011401, 0.0011721
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012688, 0.0012342
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011680, 0.0012007
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0046342, 0.0047641
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0064884, 0.0063114
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045705, 0.0044459
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0040357, 0.0041488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019607, upper bound: 0.0019082
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019607, upper bound: 0.0019082
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020428, 0.0019971
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002951, 0.0002885
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011042, 0.0011294
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011420, 0.0011681
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012645, 0.0012363
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011699, 0.0011967
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0046419, 0.0047480
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0064663, 0.0063219
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045550, 0.0044533
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0040424, 0.0041347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017647, upper bound: 0.0017018
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017785, upper bound: 0.0017014
time: 0.57 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0016781, upper bound: 0.0017160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0016781, upper bound: 0.0017160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0021061, upper bound: 0.0020520
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0021061, upper bound: 0.0020520
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0020369, upper bound: 0.0022133
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0020816, upper bound: 0.0021700
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0017346, upper bound: 0.0016491
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0017346, upper bound: 0.0016491
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0021493, upper bound: 0.0022201
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0022375, upper bound: 0.0021392
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0021493, upper bound: 0.0022201
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0022375, upper bound: 0.0021392
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0027382, upper bound: 0.0025765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0027921, upper bound: 0.0025744
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0024746, upper bound: 0.0023872
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0026344, upper bound: 0.0022876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0022569, upper bound: 0.0021564
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0022569, upper bound: 0.0021564
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0025436, upper bound: 0.0024177
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0026134, upper bound: 0.0023757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0022645, upper bound: 0.0023449
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0023234, upper bound: 0.0023362
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0024177, upper bound: 0.0022534
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0024690, upper bound: 0.0022161
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0018630, upper bound: 0.0019727
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0019422, upper bound: 0.0019049
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0019004, upper bound: 0.0019430
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0019664, upper bound: 0.0018699
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0020735, upper bound: 0.0021117
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0020752, upper bound: 0.0021042
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0025228, upper bound: 0.0026016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0025455, upper bound: 0.0025525
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0023166, upper bound: 0.0024192
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0023166, upper bound: 0.0024192
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0025334, upper bound: 0.0024220
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0025893, upper bound: 0.0024053
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0018630, upper bound: 0.0019727
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0018630, upper bound: 0.0019727
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0019004, upper bound: 0.0019430
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0019004, upper bound: 0.0019430
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0014716, upper bound: 0.0015646
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0014716, upper bound: 0.0015646
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0017159, upper bound: 0.0017527
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0017244, upper bound: 0.0017437
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0017341, upper bound: 0.0017365
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0017422, upper bound: 0.0017299
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0015644, upper bound: 0.0014748
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0015644, upper bound: 0.0014748
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0019607, upper bound: 0.0019082
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0019607, upper bound: 0.0019082
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0017647, upper bound: 0.0017018
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.66
Output dim: 8, lower bound: -0.0017785, upper bound: 0.0017014

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022780, 0.0022367
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003291, 0.0003231
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012366, 0.0012595
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012790, 0.0013026
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014101, 0.0013846
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013103, 0.0013345
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0051988, 0.0052947
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0072110, 0.0070803
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050796, 0.0049875
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0045274, 0.0046109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020452, upper bound: 0.0020469
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021011, upper bound: 0.0020000
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022720, 0.0022421
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003282, 0.0003239
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012396, 0.0012561
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012821, 0.0012992
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014064, 0.0013879
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013134, 0.0013309
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0052113, 0.0052808
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0071919, 0.0070973
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050662, 0.0049995
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0045382, 0.0045987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020445, upper bound: 0.0020469
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021011, upper bound: 0.0020000
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019822, 0.0020520
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002864, 0.0002965
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011345, 0.0010959
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011734, 0.0011335
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012270, 0.0012702
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012021, 0.0011612
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047694, 0.0046072
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0062747, 0.0064955
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0044200, 0.0045756
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041534, 0.0040122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019753, upper bound: 0.0021403
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019744, upper bound: 0.0021403
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019805, 0.0020520
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002861, 0.0002965
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011345, 0.0010950
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011734, 0.0011325
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012260, 0.0012702
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012021, 0.0011602
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047695, 0.0046032
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0062691, 0.0064957
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0044161, 0.0045757
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041535, 0.0040087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016730, upper bound: 0.0016703
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016730, upper bound: 0.0016703
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020219, 0.0020300
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002921, 0.0002933
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011223, 0.0011179
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011608, 0.0011561
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012516, 0.0012566
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011892, 0.0011844
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047183, 0.0046994
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0064002, 0.0064259
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045084, 0.0045266
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041089, 0.0040925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021202, upper bound: 0.0022179
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021473, upper bound: 0.0021969
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020751, 0.0019810
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002998, 0.0002862
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010952, 0.0011473
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011327, 0.0011866
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012845, 0.0012263
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011605, 0.0012156
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0046044, 0.0048232
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0065688, 0.0062707
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0046272, 0.0044172
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0040097, 0.0042002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017908, upper bound: 0.0017556
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017908, upper bound: 0.0017556
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020278, 0.0020323
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002930, 0.0002936
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011236, 0.0011211
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011621, 0.0011595
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012553, 0.0012580
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011905, 0.0011879
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047236, 0.0047132
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0064190, 0.0064331
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045217, 0.0045316
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041135, 0.0041045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017268, upper bound: 0.0018191
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017268, upper bound: 0.0018191
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020811, 0.0019873
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003007, 0.0002871
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010987, 0.0011506
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011363, 0.0011900
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012882, 0.0012302
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011641, 0.0012191
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0046190, 0.0048370
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0065876, 0.0062906
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0046404, 0.0044313
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0040224, 0.0042123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017908, upper bound: 0.0017556
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017908, upper bound: 0.0017556
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023361, 0.0023006
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003375, 0.0003324
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012720, 0.0012916
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013155, 0.0013358
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014461, 0.0014241
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013477, 0.0013685
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053473, 0.0054298
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0073950, 0.0072826
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0052092, 0.0051300
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046567, 0.0047285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025264, upper bound: 0.0023975
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025264, upper bound: 0.0023975
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023554, 0.0023025
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003403, 0.0003326
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012730, 0.0013023
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013166, 0.0013469
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014581, 0.0014253
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013488, 0.0013798
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053516, 0.0054747
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0074560, 0.0072884
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0052522, 0.0051341
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046604, 0.0047676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024692, upper bound: 0.0023832
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026254, upper bound: 0.0022726
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020045, 0.0020206
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002896, 0.0002919
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011172, 0.0011082
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011554, 0.0011462
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012408, 0.0012508
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011837, 0.0011742
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0046965, 0.0046590
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0063451, 0.0063962
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0044696, 0.0045056
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0040899, 0.0040572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024108, upper bound: 0.0023841
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024719, upper bound: 0.0023832
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020642, 0.0019781
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002982, 0.0002858
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010936, 0.0011413
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011311, 0.0011804
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012778, 0.0012245
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011588, 0.0012092
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0045976, 0.0047979
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0065343, 0.0062615
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0046029, 0.0044107
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0040038, 0.0041782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024604, upper bound: 0.0020858
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024139, upper bound: 0.0021247
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0025620, 0.0025150
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003701, 0.0003633
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013905, 0.0014165
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014381, 0.0014650
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015859, 0.0015568
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014733, 0.0015008
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0058455, 0.0059547
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0081099, 0.0079610
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0057127, 0.0056079
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0050905, 0.0051857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022270, upper bound: 0.0021539
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022544, upper bound: 0.0021383
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0026118, 0.0025168
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003773, 0.0003636
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013915, 0.0014440
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014391, 0.0014935
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0016168, 0.0015579
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014743, 0.0015300
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0058498, 0.0060706
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0082677, 0.0079669
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0058239, 0.0056120
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0050942, 0.0052866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022270, upper bound: 0.0021539
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022544, upper bound: 0.0021383
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0024266, 0.0023846
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003506, 0.0003445
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013184, 0.0013416
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013635, 0.0013876
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015021, 0.0014761
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013969, 0.0014215
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0055425, 0.0056402
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0076814, 0.0075484
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0054109, 0.0053173
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0048267, 0.0049117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017903, upper bound: 0.0018198
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017903, upper bound: 0.0018198
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0024281, 0.0023865
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003508, 0.0003448
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013194, 0.0013424
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013646, 0.0013884
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0015030, 0.0014773
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013980, 0.0014224
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0055468, 0.0056436
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0076860, 0.0075543
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0054142, 0.0053214
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0048304, 0.0049147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025608, upper bound: 0.0023731
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026108, upper bound: 0.0023678
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0018974, 0.0019464
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002741, 0.0002812
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010761, 0.0010490
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011130, 0.0010849
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011745, 0.0012049
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011402, 0.0011115
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0045240, 0.0044100
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0060060, 0.0061613
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0042308, 0.0043402
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0039397, 0.0038404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022372, upper bound: 0.0023390
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022584, upper bound: 0.0022984
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0018996, 0.0019474
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002744, 0.0002813
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010767, 0.0010503
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011135, 0.0010862
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011759, 0.0012055
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011408, 0.0011128
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0045262, 0.0044153
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0060132, 0.0061644
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0042358, 0.0043423
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0039417, 0.0038450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022701, upper bound: 0.0023300
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023174, upper bound: 0.0022983
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019492, 0.0018914
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002816, 0.0002732
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010457, 0.0010777
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010815, 0.0011146
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012066, 0.0011708
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011080, 0.0011418
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0043960, 0.0045304
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0061701, 0.0059870
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0043463, 0.0042174
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0038283, 0.0039453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023564, upper bound: 0.0021916
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023574, upper bound: 0.0021905
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019439, 0.0018951
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002808, 0.0002738
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010478, 0.0010747
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010837, 0.0011115
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012033, 0.0011731
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011102, 0.0011387
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0044048, 0.0045182
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0061533, 0.0059990
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0043345, 0.0042258
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0038359, 0.0039346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018483, upper bound: 0.0017104
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018483, upper bound: 0.0017104
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023945, 0.0024547
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003459, 0.0003546
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013572, 0.0013238
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0014036, 0.0013692
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014822, 0.0015195
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014380, 0.0014027
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0057054, 0.0055654
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0075795, 0.0077703
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0053392, 0.0054736
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0049686, 0.0048466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020368, upper bound: 0.0021060
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020677, upper bound: 0.0020520
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0024144, 0.0024184
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003488, 0.0003494
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013371, 0.0013349
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013829, 0.0013806
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014946, 0.0014970
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0014167, 0.0014144
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0056211, 0.0056118
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0076428, 0.0076554
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0053837, 0.0053926
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0048951, 0.0048870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020410, upper bound: 0.0020986
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020695, upper bound: 0.0020437
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023759, 0.0023715
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003432, 0.0003426
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013111, 0.0013136
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013560, 0.0013585
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014707, 0.0014680
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013892, 0.0013918
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0055120, 0.0055222
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0075207, 0.0075069
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0052978, 0.0052880
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0048001, 0.0048090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020368, upper bound: 0.0021060
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020410, upper bound: 0.0020986
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023739, 0.0023517
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003430, 0.0003397
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0013002, 0.0013125
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013447, 0.0013574
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014695, 0.0014557
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013776, 0.0013906
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0054659, 0.0055177
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0075146, 0.0074441
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0052934, 0.0052438
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0047600, 0.0048050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022473, upper bound: 0.0024005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023962, upper bound: 0.0022773
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019600, 0.0020489
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002832, 0.0002960
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011328, 0.0010836
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011716, 0.0011207
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012132, 0.0012683
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012002, 0.0011481
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047622, 0.0045555
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0062042, 0.0064857
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0043704, 0.0045686
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041471, 0.0039671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014309, upper bound: 0.0014543
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014309, upper bound: 0.0014543
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019998, 0.0020351
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002889, 0.0002940
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011252, 0.0011056
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011637, 0.0011435
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012379, 0.0012598
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011922, 0.0011715
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047302, 0.0046480
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0063302, 0.0064421
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0044591, 0.0045379
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041192, 0.0040477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022652, upper bound: 0.0024135
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023110, upper bound: 0.0023929
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0018011, 0.0017922
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002602, 0.0002589
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009908, 0.0009958
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010248, 0.0010299
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011149, 0.0011094
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010498, 0.0010551
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0041655, 0.0041864
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0057015, 0.0056730
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0040162, 0.0039962
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0036275, 0.0036457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012366, upper bound: 0.0012245
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012366, upper bound: 0.0012245
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0018121, 0.0017895
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002618, 0.0002585
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009894, 0.0010019
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010233, 0.0010362
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011217, 0.0011077
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010483, 0.0010615
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0041594, 0.0042119
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0057362, 0.0056647
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0040407, 0.0039903
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0036222, 0.0036679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019027, upper bound: 0.0017808
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019027, upper bound: 0.0017808
time: 0.59 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0020452, upper bound: 0.0020469
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0021011, upper bound: 0.0020000
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0020445, upper bound: 0.0020469
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0021011, upper bound: 0.0020000
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0019753, upper bound: 0.0021403
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0019744, upper bound: 0.0021403
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0016730, upper bound: 0.0016703
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0016730, upper bound: 0.0016703
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0021202, upper bound: 0.0022179
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0021473, upper bound: 0.0021969
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0017908, upper bound: 0.0017556
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0017908, upper bound: 0.0017556
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0017268, upper bound: 0.0018191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0017268, upper bound: 0.0018191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0017908, upper bound: 0.0017556
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0017908, upper bound: 0.0017556
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0025264, upper bound: 0.0023975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0025264, upper bound: 0.0023975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0024692, upper bound: 0.0023832
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0026254, upper bound: 0.0022726
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0024108, upper bound: 0.0023841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0024719, upper bound: 0.0023832
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0024604, upper bound: 0.0020858
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0024139, upper bound: 0.0021247
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0022270, upper bound: 0.0021539
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0022544, upper bound: 0.0021383
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0022270, upper bound: 0.0021539
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0022544, upper bound: 0.0021383
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0017903, upper bound: 0.0018198
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0017903, upper bound: 0.0018198
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0025608, upper bound: 0.0023731
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0026108, upper bound: 0.0023678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0022372, upper bound: 0.0023390
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0022584, upper bound: 0.0022984
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0022701, upper bound: 0.0023300
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0023174, upper bound: 0.0022983
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0023564, upper bound: 0.0021916
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0023574, upper bound: 0.0021905
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0018483, upper bound: 0.0017104
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0018483, upper bound: 0.0017104
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0020368, upper bound: 0.0021060
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0020677, upper bound: 0.0020520
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0020410, upper bound: 0.0020986
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0020695, upper bound: 0.0020437
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0020368, upper bound: 0.0021060
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0020410, upper bound: 0.0020986
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0022473, upper bound: 0.0024005
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0023962, upper bound: 0.0022773
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0014309, upper bound: 0.0014543
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0014309, upper bound: 0.0014543
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0022652, upper bound: 0.0024135
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0023110, upper bound: 0.0023929
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0012366, upper bound: 0.0012245
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0012366, upper bound: 0.0012245
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0019027, upper bound: 0.0017808
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.82
Output dim: 8, lower bound: -0.0019027, upper bound: 0.0017808

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020246, 0.0019831
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002925, 0.0002865
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010964, 0.0011194
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011340, 0.0011577
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012533, 0.0012276
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011617, 0.0011860
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0046093, 0.0047058
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0064088, 0.0062775
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045145, 0.0044220
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0040140, 0.0040980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020826, upper bound: 0.0019976
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020990, upper bound: 0.0019830
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020192, 0.0019885
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002917, 0.0002873
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010994, 0.0011163
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011370, 0.0011546
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012499, 0.0012309
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011648, 0.0011828
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0046218, 0.0046931
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0063916, 0.0062945
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0045024, 0.0044339
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0040248, 0.0040869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020826, upper bound: 0.0019976
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020990, upper bound: 0.0019830
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019793, 0.0020443
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002860, 0.0002953
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011302, 0.0010943
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011690, 0.0011318
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012252, 0.0012655
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011976, 0.0011595
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047515, 0.0046005
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0062654, 0.0064712
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0044135, 0.0045584
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041379, 0.0040063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015689, upper bound: 0.0016554
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015689, upper bound: 0.0016554
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019745, 0.0020504
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002853, 0.0002962
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011336, 0.0010917
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0011724, 0.0011291
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012223, 0.0012692
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012011, 0.0011567
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0047657, 0.0045894
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0062503, 0.0064905
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0044029, 0.0045720
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0041502, 0.0039966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015689, upper bound: 0.0016554
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015689, upper bound: 0.0016554
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017539, 0.0017652
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002534, 0.0002550
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009760, 0.0009697
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010094, 0.0010029
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010857, 0.0010927
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010341, 0.0010274
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0041029, 0.0040766
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055519, 0.0055878
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039109, 0.0039362
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0035730, 0.0035500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017073, upper bound: 0.0018172
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017073, upper bound: 0.0018172
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017583, 0.0017620
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002540, 0.0002546
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009742, 0.0009721
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010075, 0.0010054
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010884, 0.0010907
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010322, 0.0010300
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0040954, 0.0040868
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055659, 0.0055776
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039207, 0.0039290
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0035665, 0.0035590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017714, upper bound: 0.0018504
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017714, upper bound: 0.0018504
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022892, 0.0022582
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003307, 0.0003262
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012485, 0.0012656
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012913, 0.0013090
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014171, 0.0013979
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013229, 0.0013410
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0052487, 0.0053207
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0072464, 0.0071483
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0051045, 0.0050354
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0045708, 0.0046335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014263, upper bound: 0.0014067
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014263, upper bound: 0.0014067
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023361, 0.0022537
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003375, 0.0003256
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012460, 0.0012916
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012887, 0.0013358
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014461, 0.0013951
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013202, 0.0013685
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0052382, 0.0054298
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0073950, 0.0071340
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0052092, 0.0050253
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0045617, 0.0047285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022204, upper bound: 0.0022288
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023810, upper bound: 0.0021288
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017580, 0.0017495
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002540, 0.0002527
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009672, 0.0009719
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010004, 0.0010052
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010882, 0.0010830
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010248, 0.0010298
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0040663, 0.0040860
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055647, 0.0055379
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039199, 0.0039010
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0035411, 0.0035582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022798, upper bound: 0.0022285
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022798, upper bound: 0.0022285
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0018011, 0.0017050
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002602, 0.0002463
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009427, 0.0009958
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009749, 0.0010299
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011149, 0.0010554
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009988, 0.0010551
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039629, 0.0041864
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0057015, 0.0053971
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0040162, 0.0038019
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034511, 0.0036457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015186, upper bound: 0.0014622
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015186, upper bound: 0.0014622
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017347, 0.0017431
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002506, 0.0002518
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009637, 0.0009591
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009967, 0.0009919
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010738, 0.0010790
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010211, 0.0010162
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0040516, 0.0040318
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0054910, 0.0055179
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0038680, 0.0038869
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0035283, 0.0035111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022419, upper bound: 0.0021714
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022084, upper bound: 0.0022269
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017591, 0.0017508
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002541, 0.0002529
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009680, 0.0009726
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010011, 0.0010059
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010889, 0.0010838
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010256, 0.0010305
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0040694, 0.0040887
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055684, 0.0055421
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039225, 0.0039040
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0035438, 0.0035606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020599, upper bound: 0.0020407
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020599, upper bound: 0.0020407
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020003, 0.0018824
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002890, 0.0002720
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010407, 0.0011059
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010764, 0.0011438
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012382, 0.0011652
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011027, 0.0011718
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0043752, 0.0046492
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0063318, 0.0059586
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0044603, 0.0041974
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0038101, 0.0040487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021148, upper bound: 0.0018592
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021148, upper bound: 0.0018592
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019686, 0.0019039
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002844, 0.0002751
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010526, 0.0010884
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010887, 0.0011256
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012186, 0.0011786
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0011153, 0.0011532
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0044253, 0.0045755
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0062314, 0.0060268
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0043895, 0.0042454
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0038537, 0.0039845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016714, upper bound: 0.0015510
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016714, upper bound: 0.0015510
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023433, 0.0023010
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003385, 0.0003324
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012722, 0.0012955
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013157, 0.0013399
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014505, 0.0014244
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013479, 0.0013727
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053482, 0.0054464
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0074175, 0.0072837
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0052251, 0.0051308
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046574, 0.0047430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019468, upper bound: 0.0020012
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020853, upper bound: 0.0019263
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023465, 0.0022963
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003390, 0.0003317
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012695, 0.0012973
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013130, 0.0013417
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014525, 0.0014214
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013451, 0.0013746
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053371, 0.0054539
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0074277, 0.0072687
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0052322, 0.0051202
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046478, 0.0047495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019969, upper bound: 0.0019885
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021095, upper bound: 0.0018848
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023932, 0.0022974
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003457, 0.0003319
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012702, 0.0013231
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013137, 0.0013685
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014814, 0.0014222
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013458, 0.0014019
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053399, 0.0055625
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0075756, 0.0072724
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0053364, 0.0051229
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046502, 0.0048440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019468, upper bound: 0.0020012
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020853, upper bound: 0.0019263
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023964, 0.0022981
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003462, 0.0003320
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012706, 0.0013249
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013141, 0.0013703
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014834, 0.0014226
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013462, 0.0014038
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053414, 0.0055699
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0075858, 0.0072746
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0053436, 0.0051244
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046515, 0.0048506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021881, upper bound: 0.0021324
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022488, upper bound: 0.0021030
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022575, 0.0022119
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003261, 0.0003196
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012229, 0.0012481
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012648, 0.0012909
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0013975, 0.0013692
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012957, 0.0013225
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0051411, 0.0052472
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0071462, 0.0070018
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050339, 0.0049322
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0044771, 0.0045695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018091, upper bound: 0.0017796
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018091, upper bound: 0.0017796
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022776, 0.0022159
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003290, 0.0003201
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012251, 0.0012592
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012671, 0.0013024
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014099, 0.0013717
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012981, 0.0013342
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0051504, 0.0052938
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0072097, 0.0070144
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050787, 0.0049411
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0044852, 0.0046101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018336, upper bound: 0.0017787
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018336, upper bound: 0.0017787
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016625, 0.0017209
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002402, 0.0002486
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009515, 0.0009192
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009840, 0.0009506
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010291, 0.0010653
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010081, 0.0009739
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039999, 0.0038641
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0052626, 0.0054475
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037071, 0.0038374
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034833, 0.0033650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017073, upper bound: 0.0018172
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017073, upper bound: 0.0018172
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016597, 0.0017116
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002398, 0.0002473
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009463, 0.0009176
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009787, 0.0009490
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010274, 0.0010595
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010026, 0.0009722
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039781, 0.0038576
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0052537, 0.0054179
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037008, 0.0038165
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034643, 0.0033593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021983, upper bound: 0.0022363
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021984, upper bound: 0.0022361
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016648, 0.0017154
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002405, 0.0002478
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009484, 0.0009204
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009809, 0.0009519
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010305, 0.0010619
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010049, 0.0009752
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039871, 0.0038694
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0052698, 0.0054301
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037121, 0.0038251
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034722, 0.0033696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022086, upper bound: 0.0022677
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022086, upper bound: 0.0022677
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016782, 0.0017125
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002424, 0.0002474
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009468, 0.0009278
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009792, 0.0009596
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010388, 0.0010601
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010032, 0.0009831
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039804, 0.0039005
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0053122, 0.0054209
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037420, 0.0038186
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034663, 0.0033968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017776, upper bound: 0.0017691
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017776, upper bound: 0.0017691
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019317, 0.0018701
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002791, 0.0002702
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010339, 0.0010680
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010693, 0.0011046
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011958, 0.0011576
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010955, 0.0011316
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0043466, 0.0044898
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0061147, 0.0059197
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0043073, 0.0041699
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0037852, 0.0039099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016530, upper bound: 0.0015945
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016530, upper bound: 0.0015945
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019319, 0.0018739
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002791, 0.0002707
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010360, 0.0010681
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010715, 0.0011047
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011959, 0.0011600
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010977, 0.0011317
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0043554, 0.0044902
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0061152, 0.0059316
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0043077, 0.0041784
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0037929, 0.0039102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023215, upper bound: 0.0021843
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023512, upper bound: 0.0021284
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022644, 0.0023404
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003271, 0.0003381
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012940, 0.0012519
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013383, 0.0012948
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014017, 0.0014488
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013710, 0.0013265
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0054398, 0.0052630
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0071678, 0.0074086
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050492, 0.0052187
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0047372, 0.0045833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018078, upper bound: 0.0019668
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018959, upper bound: 0.0018802
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022624, 0.0023246
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003269, 0.0003358
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012852, 0.0012508
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013293, 0.0012937
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014005, 0.0014390
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013618, 0.0013253
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0054031, 0.0052585
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0071617, 0.0073586
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050448, 0.0051835
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0047053, 0.0045794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018266, upper bound: 0.0019147
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019276, upper bound: 0.0018265
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022844, 0.0023073
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003300, 0.0003333
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012756, 0.0012630
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013193, 0.0013062
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014141, 0.0014282
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013516, 0.0013382
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053627, 0.0053095
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0072311, 0.0073035
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050937, 0.0051448
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046701, 0.0046237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018078, upper bound: 0.0019599
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019027, upper bound: 0.0018786
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022820, 0.0022883
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003297, 0.0003306
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012652, 0.0012616
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013085, 0.0013048
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014126, 0.0014165
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013405, 0.0013368
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053187, 0.0053039
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0072235, 0.0072437
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050884, 0.0051026
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046318, 0.0046189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018266, upper bound: 0.0019056
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019311, upper bound: 0.0018252
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023110, 0.0023381
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003339, 0.0003378
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012927, 0.0012777
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013369, 0.0013214
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014305, 0.0014473
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013696, 0.0013538
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0054343, 0.0053713
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0073153, 0.0074010
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0051530, 0.0052134
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0047324, 0.0046776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018078, upper bound: 0.0019668
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018959, upper bound: 0.0018802
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0023309, 0.0023071
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003368, 0.0003333
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0012755, 0.0012887
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0013192, 0.0013329
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014429, 0.0014281
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0013515, 0.0013655
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0053622, 0.0054177
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0073785, 0.0073029
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0051976, 0.0051443
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0046697, 0.0047180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018078, upper bound: 0.0019599
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019027, upper bound: 0.0018786
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017643, 0.0017889
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002549, 0.0002584
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009890, 0.0009754
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010229, 0.0010089
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010921, 0.0011074
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010479, 0.0010335
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0041579, 0.0041008
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055849, 0.0056627
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039341, 0.0039889
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0036209, 0.0035711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018266, upper bound: 0.0019147
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018266, upper bound: 0.0019056
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0018093, 0.0017490
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002614, 0.0002527
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009670, 0.0010003
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010001, 0.0010346
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011200, 0.0010827
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010246, 0.0010599
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0040652, 0.0042053
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0057273, 0.0055364
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0040344, 0.0038999
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0035401, 0.0036622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019276, upper bound: 0.0018265
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019311, upper bound: 0.0018252
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017617, 0.0017978
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002545, 0.0002597
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009940, 0.0009740
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010280, 0.0010073
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010905, 0.0011129
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010531, 0.0010320
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0041786, 0.0040946
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055765, 0.0056909
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039282, 0.0040088
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0036389, 0.0035658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0013864, upper bound: 0.0014484
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0013864, upper bound: 0.0014484
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017735, 0.0017967
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002562, 0.0002596
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009934, 0.0009805
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010274, 0.0010141
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010979, 0.0011122
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010525, 0.0010389
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0041760, 0.0041222
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0056141, 0.0056874
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039547, 0.0040063
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0036367, 0.0035898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014253, upper bound: 0.0014368
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014253, upper bound: 0.0014368
time: 0.58 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0020826, upper bound: 0.0019976
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0020990, upper bound: 0.0019830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0020826, upper bound: 0.0019976
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0020990, upper bound: 0.0019830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0015689, upper bound: 0.0016554
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0015689, upper bound: 0.0016554
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0015689, upper bound: 0.0016554
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0015689, upper bound: 0.0016554
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0017073, upper bound: 0.0018172
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0017073, upper bound: 0.0018172
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0017714, upper bound: 0.0018504
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0017714, upper bound: 0.0018504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0014263, upper bound: 0.0014067
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0014263, upper bound: 0.0014067
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0022204, upper bound: 0.0022288
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0023810, upper bound: 0.0021288
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0022798, upper bound: 0.0022285
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0022798, upper bound: 0.0022285
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0015186, upper bound: 0.0014622
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0015186, upper bound: 0.0014622
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0022419, upper bound: 0.0021714
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0022084, upper bound: 0.0022269
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0020599, upper bound: 0.0020407
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0020599, upper bound: 0.0020407
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0021148, upper bound: 0.0018592
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0021148, upper bound: 0.0018592
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0016714, upper bound: 0.0015510
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0016714, upper bound: 0.0015510
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0019468, upper bound: 0.0020012
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0020853, upper bound: 0.0019263
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0019969, upper bound: 0.0019885
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0021095, upper bound: 0.0018848
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0019468, upper bound: 0.0020012
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0020853, upper bound: 0.0019263
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0021881, upper bound: 0.0021324
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0022488, upper bound: 0.0021030
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018091, upper bound: 0.0017796
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018091, upper bound: 0.0017796
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018336, upper bound: 0.0017787
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018336, upper bound: 0.0017787
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0017073, upper bound: 0.0018172
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0017073, upper bound: 0.0018172
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0021983, upper bound: 0.0022363
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0021984, upper bound: 0.0022361
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0022086, upper bound: 0.0022677
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0022086, upper bound: 0.0022677
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0017776, upper bound: 0.0017691
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0017776, upper bound: 0.0017691
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0016530, upper bound: 0.0015945
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0016530, upper bound: 0.0015945
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0023215, upper bound: 0.0021843
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0023512, upper bound: 0.0021284
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018078, upper bound: 0.0019668
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018959, upper bound: 0.0018802
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018266, upper bound: 0.0019147
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0019276, upper bound: 0.0018265
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018078, upper bound: 0.0019599
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0019027, upper bound: 0.0018786
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018266, upper bound: 0.0019056
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0019311, upper bound: 0.0018252
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018078, upper bound: 0.0019668
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018959, upper bound: 0.0018802
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018078, upper bound: 0.0019599
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0019027, upper bound: 0.0018786
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018266, upper bound: 0.0019147
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0018266, upper bound: 0.0019056
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0019276, upper bound: 0.0018265
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0019311, upper bound: 0.0018252
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0013864, upper bound: 0.0014484
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0013864, upper bound: 0.0014484
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0014253, upper bound: 0.0014368
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.79
Output dim: 8, lower bound: -0.0014253, upper bound: 0.0014368

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017556, 0.0017234
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002536, 0.0002490
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009528, 0.0009706
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009855, 0.0010039
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010867, 0.0010668
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010096, 0.0010284
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0040057, 0.0040804
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055572, 0.0054554
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039146, 0.0038429
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034883, 0.0035534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017587, 0.0017141
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002541, 0.0002476
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009477, 0.0009723
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009801, 0.0010056
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010886, 0.0010610
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010041, 0.0010302
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039840, 0.0040876
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055670, 0.0054258
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039215, 0.0038221
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034694, 0.0035597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012765, upper bound: 0.0012148
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012765, upper bound: 0.0012148
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017501, 0.0017298
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002528, 0.0002499
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009564, 0.0009676
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009891, 0.0010007
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010834, 0.0010708
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010133, 0.0010252
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0040206, 0.0040678
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055399, 0.0054757
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039024, 0.0038572
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0035013, 0.0035424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017528, 0.0017194
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002532, 0.0002484
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009506, 0.0009691
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009832, 0.0010022
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010850, 0.0010644
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010072, 0.0010268
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039964, 0.0040739
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055483, 0.0054428
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039083, 0.0038340
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034803, 0.0035477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016675, upper bound: 0.0015345
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016675, upper bound: 0.0015345
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017387, 0.0017012
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002512, 0.0002458
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009405, 0.0009613
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009728, 0.0009942
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010763, 0.0010531
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009966, 0.0010185
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039540, 0.0040411
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055037, 0.0053850
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0038769, 0.0037933
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034433, 0.0035192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019434, upper bound: 0.0019483
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019223, upper bound: 0.0019670
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017947, 0.0016618
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002593, 0.0002401
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009188, 0.0009922
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009503, 0.0010262
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011109, 0.0010287
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009735, 0.0010513
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0038626, 0.0041713
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0056810, 0.0052605
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0040018, 0.0037056
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0033637, 0.0036326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017166, 0.0017178
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002480, 0.0002482
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009497, 0.0009491
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009823, 0.0009816
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010626, 0.0010634
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010063, 0.0010056
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039927, 0.0039899
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0054340, 0.0054378
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0038278, 0.0038305
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034770, 0.0034746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012522, upper bound: 0.0012450
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012522, upper bound: 0.0012450
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017580, 0.0017082
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002540, 0.0002468
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009444, 0.0009719
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009767, 0.0010052
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010882, 0.0010574
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010006, 0.0010298
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039702, 0.0040860
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055647, 0.0054071
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0039199, 0.0038089
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034574, 0.0035582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012522, upper bound: 0.0012450
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012522, upper bound: 0.0012450
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016677, 0.0016512
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002409, 0.0002385
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009129, 0.0009220
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009442, 0.0009536
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010323, 0.0010221
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009673, 0.0009769
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0038378, 0.0038761
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0052789, 0.0052267
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037186, 0.0036818
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0033421, 0.0033755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019434, upper bound: 0.0019483
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019434, upper bound: 0.0019483
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016427, 0.0016869
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002373, 0.0002437
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009326, 0.0009082
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009646, 0.0009393
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010169, 0.0010442
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009882, 0.0009623
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039207, 0.0038181
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0051999, 0.0053397
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0036629, 0.0037614
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034144, 0.0033249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019223, upper bound: 0.0019670
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019223, upper bound: 0.0019670
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019527, 0.0018341
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002821, 0.0002650
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010140, 0.0010796
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010487, 0.0011166
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012087, 0.0011353
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010744, 0.0011439
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0042628, 0.0045385
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0061811, 0.0058056
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0043541, 0.0040896
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0037123, 0.0039524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020860, upper bound: 0.0018566
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021123, upper bound: 0.0018491
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0020003, 0.0018348
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002890, 0.0002651
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010144, 0.0011059
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010491, 0.0011438
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012382, 0.0011358
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010748, 0.0011718
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0042645, 0.0046492
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0063318, 0.0058079
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0044603, 0.0040912
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0037137, 0.0040487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020860, upper bound: 0.0018566
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021123, upper bound: 0.0018491
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019177, 0.0018137
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002771, 0.0002620
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010028, 0.0010602
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010371, 0.0010966
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011871, 0.0011227
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010625, 0.0011234
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0042156, 0.0044572
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0060704, 0.0057413
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0042761, 0.0040443
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0036712, 0.0038816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020458, upper bound: 0.0019205
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020794, upper bound: 0.0018599
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019179, 0.0018090
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002771, 0.0002613
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010001, 0.0010604
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010344, 0.0010967
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0011872, 0.0011198
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010597, 0.0011235
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0042046, 0.0044577
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0060710, 0.0057263
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0042766, 0.0040337
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0036616, 0.0038820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020461, upper bound: 0.0018789
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021038, upper bound: 0.0018513
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0019602, 0.0018102
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002832, 0.0002615
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0010008, 0.0010837
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0010351, 0.0011208
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0012134, 0.0011205
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0010604, 0.0011483
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0042073, 0.0045559
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0062048, 0.0057300
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0043708, 0.0040364
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0036639, 0.0039675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020458, upper bound: 0.0019205
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020794, upper bound: 0.0018599
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022611, 0.0021642
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003267, 0.0003127
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011965, 0.0012501
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012375, 0.0012929
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0013997, 0.0013397
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012678, 0.0013246
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0050302, 0.0052554
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0071575, 0.0068508
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050419, 0.0048258
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0043806, 0.0045767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019291, upper bound: 0.0019827
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020461, upper bound: 0.0018789
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0022783, 0.0021658
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0003291, 0.0003129
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0011974, 0.0012596
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0012384, 0.0013027
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0014103, 0.0013407
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0012687, 0.0013346
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0050339, 0.0052953
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0072118, 0.0068557
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0050801, 0.0048293
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0043837, 0.0046114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019911, upper bound: 0.0019530
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021038, upper bound: 0.0018513
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016414, 0.0016940
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002371, 0.0002447
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009365, 0.0009075
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009686, 0.0009385
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010160, 0.0010486
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009923, 0.0009615
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039372, 0.0038150
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0051957, 0.0053622
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0036599, 0.0037772
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034287, 0.0033223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015681, upper bound: 0.0016137
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015681, upper bound: 0.0016137
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016379, 0.0016932
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002366, 0.0002446
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009362, 0.0009055
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009682, 0.0009365
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010139, 0.0010481
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009919, 0.0009595
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039356, 0.0038068
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0051846, 0.0053599
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0036521, 0.0037756
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034273, 0.0033151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015681, upper bound: 0.0016137
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015681, upper bound: 0.0016137
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016465, 0.0016975
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002379, 0.0002452
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009385, 0.0009103
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009706, 0.0009415
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010192, 0.0010508
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009944, 0.0009645
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039454, 0.0038268
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0052118, 0.0053732
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0036713, 0.0037850
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034358, 0.0033326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015674, upper bound: 0.0016333
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015674, upper bound: 0.0016333
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016469, 0.0016971
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002379, 0.0002452
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009383, 0.0009105
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009704, 0.0009417
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010194, 0.0010505
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009942, 0.0009647
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0039446, 0.0038278
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0052131, 0.0053722
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0036722, 0.0037843
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0034351, 0.0033334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019208, upper bound: 0.0020138
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019208, upper bound: 0.0020138
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016962, 0.0016545
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002450, 0.0002390
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009147, 0.0009378
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009461, 0.0009699
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010500, 0.0010242
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009692, 0.0009936
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0038456, 0.0039424
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0053692, 0.0052374
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037822, 0.0036893
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0033489, 0.0034332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016209, upper bound: 0.0015896
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016209, upper bound: 0.0015896
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016959, 0.0016382
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002450, 0.0002367
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0009057, 0.0009376
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009367, 0.0009697
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010498, 0.0010141
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009596, 0.0009935
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0038076, 0.0039418
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0053684, 0.0051856
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037816, 0.0036528
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0033158, 0.0034327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016480, upper bound: 0.0015491
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016480, upper bound: 0.0015491
time: 0.60 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012765, upper bound: 0.0012148
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012765, upper bound: 0.0012148
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0016675, upper bound: 0.0015345
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0016675, upper bound: 0.0015345
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0019434, upper bound: 0.0019483
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0019223, upper bound: 0.0019670
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012690, upper bound: 0.0012209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012522, upper bound: 0.0012450
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012522, upper bound: 0.0012450
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012522, upper bound: 0.0012450
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0012522, upper bound: 0.0012450
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0019434, upper bound: 0.0019483
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0019434, upper bound: 0.0019483
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0019223, upper bound: 0.0019670
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0019223, upper bound: 0.0019670
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0020860, upper bound: 0.0018566
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0021123, upper bound: 0.0018491
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0020860, upper bound: 0.0018566
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0021123, upper bound: 0.0018491
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0020458, upper bound: 0.0019205
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0020794, upper bound: 0.0018599
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0020461, upper bound: 0.0018789
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0021038, upper bound: 0.0018513
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0020458, upper bound: 0.0019205
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0020794, upper bound: 0.0018599
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0019291, upper bound: 0.0019827
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0020461, upper bound: 0.0018789
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0019911, upper bound: 0.0019530
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0021038, upper bound: 0.0018513
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0015681, upper bound: 0.0016137
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0015681, upper bound: 0.0016137
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0015681, upper bound: 0.0016137
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0015681, upper bound: 0.0016137
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0015674, upper bound: 0.0016333
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0015674, upper bound: 0.0016333
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0019208, upper bound: 0.0020138
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0019208, upper bound: 0.0020138
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0016209, upper bound: 0.0015896
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0016209, upper bound: 0.0015896
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0016480, upper bound: 0.0015491
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.86
Output dim: 8, lower bound: -0.0016480, upper bound: 0.0015491

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016929, 0.0015776
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002446, 0.0002279
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0008722, 0.0009359
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009021, 0.0009680
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010479, 0.0009766
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009242, 0.0009917
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0036668, 0.0039347
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0053587, 0.0049939
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037748, 0.0035178
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0031933, 0.0034265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 169
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 143

Time for candidate selection: 4.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019571, upper bound: 0.0018392
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020686, upper bound: 0.0017889
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016996, 0.0015742
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002455, 0.0002274
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0008704, 0.0009397
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009002, 0.0009719
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010521, 0.0009745
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009222, 0.0009956
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0036590, 0.0039504
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0053801, 0.0049832
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037899, 0.0035103
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0031864, 0.0034402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 169
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 3.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018532, upper bound: 0.0016699
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018532, upper bound: 0.0016699
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017342, 0.0015704
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002505, 0.0002269
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0008682, 0.0009588
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0008980, 0.0009916
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010735, 0.0009721
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009199, 0.0010159
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0036500, 0.0040307
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0054895, 0.0049710
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0038669, 0.0035017
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0031786, 0.0035101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 169
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 96

Time for candidate selection: 4.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012687, upper bound: 0.0011937
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012687, upper bound: 0.0011937
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017409, 0.0015750
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002515, 0.0002275
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0008708, 0.0009625
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009006, 0.0009955
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010777, 0.0009749
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009226, 0.0010198
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0036607, 0.0040464
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0055109, 0.0049855
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0038820, 0.0035119
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0031879, 0.0035238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 169
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 156

Time for candidate selection: 4.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020396, upper bound: 0.0017729
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020395, upper bound: 0.0017727
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016856, 0.0015792
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002435, 0.0002281
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0008731, 0.0009319
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009030, 0.0009638
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010434, 0.0009775
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009251, 0.0009874
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0036705, 0.0039177
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0053355, 0.0049989
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037585, 0.0035213
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0031964, 0.0034117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 169
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 4.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014416, upper bound: 0.0013134
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014416, upper bound: 0.0013134
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0016919, 0.0015745
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002444, 0.0002275
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0008705, 0.0009354
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009003, 0.0009675
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010473, 0.0009746
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009223, 0.0009911
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0036595, 0.0039325
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0053557, 0.0049839
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0037727, 0.0035108
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0031868, 0.0034246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 169
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 156

Time for candidate selection: 4.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0013908, upper bound: 0.0013080
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0013908, upper bound: 0.0013080
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017269, 0.0015756
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002495, 0.0002276
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0008711, 0.0009547
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009010, 0.0009874
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010690, 0.0009753
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009230, 0.0010116
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0036622, 0.0040137
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0054663, 0.0049876
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0038506, 0.0035134
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0031892, 0.0034953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 169

Time for candidate selection: 4.03 seconds

### Candidate
type: RSZ, layer: 3, pos: 225

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019827, upper bound: 0.0017840
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019879, upper bound: 0.0017848
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0058811, 0.0088492, 0.0058811, 0.0088492, -0.0017332, 0.0015763
1: 0.0021720, 0.0026008, 0.0021720, 0.0026008, -0.0002504, 0.0002277
2: 0.0094674, 0.0111083, 0.0094674, 0.0111083, -0.0008715, 0.0009583
3: -0.0048889, -0.0031917, -0.0048889, -0.0031917, -0.0009013, 0.0009911
4: -0.0005818, 0.0012555, -0.0005818, 0.0012555, -0.0010729, 0.0009758
5: 0.0029252, 0.0046639, 0.0029252, 0.0046639, -0.0009234, 0.0010153
6: -0.0106939, -0.0037953, -0.0106939, -0.0037953, -0.0036638, 0.0040285
7: 0.0026122, 0.0120075, 0.0026122, 0.0120075, -0.0054865, 0.0049897
8: 0.9910539, 0.9976722, 0.9910539, 0.9976722, -0.0038648, 0.0035149
9: -0.0137743, -0.0077667, -0.0137743, -0.0077667, -0.0031906, 0.0035082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 156
type: RSZ, layer: 3, pos: 134
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 169
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 143
type: RSZ, layer: 3, pos: 225

Time for candidate selection: 3.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014420, upper bound: 0.0013109
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014420, upper bound: 0.0013109
time: 0.63 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 6.71 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0019571, upper bound: 0.0018392
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0020686, upper bound: 0.0017889
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0018532, upper bound: 0.0016699
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0018532, upper bound: 0.0016699
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0012687, upper bound: 0.0011937
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0012687, upper bound: 0.0011937
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0020396, upper bound: 0.0017729
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0020395, upper bound: 0.0017727
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0014416, upper bound: 0.0013134
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0014416, upper bound: 0.0013134
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0013908, upper bound: 0.0013080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0013908, upper bound: 0.0013080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0019827, upper bound: 0.0017840
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0019879, upper bound: 0.0017848
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0014420, upper bound: 0.0013109
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 6.71
Output dim: 8, lower bound: -0.0014420, upper bound: 0.0013109

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.28 + 597.02 = 600.30 seconds
