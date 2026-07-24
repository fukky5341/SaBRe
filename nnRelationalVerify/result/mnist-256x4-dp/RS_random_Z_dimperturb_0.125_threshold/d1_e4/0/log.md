## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 4.776e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0002341, 0.0002341)
1: (-0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000379, 0.0000379)
2: (0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002924, 0.0002924)
3: (1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000707, 0.0000707)
4: (-0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000442, 0.0000442)
5: (0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001786, 0.0001786)
6: (-0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000108, 0.0000108)
7: (-0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0004399, 0.0004399)
8: (-0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0004545, 0.0004545)
9: (-0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0002115, 0.0002115)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.44 + 1.26 = 2.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000545

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000524
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000543
time: 0.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 3, lower bound: -0.0000543, upper bound: 0.0000524
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 3, lower bound: -0.0000524, upper bound: 0.0000543

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0002068, 0.0002050
1: -0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000357, 0.0000357
2: 0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002597, 0.0002575
3: 1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000696, 0.0000692
4: -0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000392, 0.0000395
5: 0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001579, 0.0001565
6: -0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000090, 0.0000089
7: -0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0003766, 0.0003806
8: -0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0004058, 0.0004087
9: -0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0001912, 0.0001903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000500, upper bound: 0.0000499
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000482
time: 0.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0002050, 0.0002068
1: -0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000357, 0.0000357
2: 0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002575, 0.0002597
3: 1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000692, 0.0000696
4: -0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000395, 0.0000392
5: 0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001565, 0.0001579
6: -0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000089, 0.0000090
7: -0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0003806, 0.0003766
8: -0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0004087, 0.0004058
9: -0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0001903, 0.0001912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000482, upper bound: 0.0000519
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000499, upper bound: 0.0000500
time: 0.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 3, lower bound: -0.0000500, upper bound: 0.0000499
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 3, lower bound: -0.0000519, upper bound: 0.0000482
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 3, lower bound: -0.0000482, upper bound: 0.0000519
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 3, lower bound: -0.0000499, upper bound: 0.0000500

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0001868, 0.0001796
1: -0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000299, 0.0000282
2: 0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002328, 0.0002234
3: 1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000588, 0.0000596
4: -0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000336, 0.0000351
5: 0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001425, 0.0001370
6: -0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000089, 0.0000088
7: -0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0003430, 0.0003529
8: -0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0003439, 0.0003601
9: -0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0001666, 0.0001590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000471, upper bound: 0.0000456
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000460, upper bound: 0.0000476
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0001814, 0.0001851
1: -0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000282, 0.0000300
2: 0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002256, 0.0002308
3: 1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000601, 0.0000584
4: -0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000348, 0.0000339
5: 0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001383, 0.0001412
6: -0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000089, 0.0000088
7: -0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0003482, 0.0003470
8: -0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0003579, 0.0003468
9: -0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0001598, 0.0001659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000493, upper bound: 0.0000441
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000480, upper bound: 0.0000456
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0001851, 0.0001814
1: -0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000300, 0.0000282
2: 0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002308, 0.0002256
3: 1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000584, 0.0000601
4: -0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000339, 0.0000348
5: 0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001412, 0.0001383
6: -0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000088, 0.0000089
7: -0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0003470, 0.0003482
8: -0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0003468, 0.0003579
9: -0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0001659, 0.0001598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000456, upper bound: 0.0000480
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000441, upper bound: 0.0000493
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0001796, 0.0001868
1: -0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000282, 0.0000299
2: 0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002234, 0.0002328
3: 1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000596, 0.0000588
4: -0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000351, 0.0000336
5: 0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001370, 0.0001425
6: -0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000088, 0.0000089
7: -0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0003529, 0.0003430
8: -0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0003601, 0.0003439
9: -0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0001590, 0.0001666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000476, upper bound: 0.0000460
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000456, upper bound: 0.0000471
time: 0.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 3, lower bound: -0.0000471, upper bound: 0.0000456
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 3, lower bound: -0.0000460, upper bound: 0.0000476
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 3, lower bound: -0.0000493, upper bound: 0.0000441
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 3, lower bound: -0.0000480, upper bound: 0.0000456
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 3, lower bound: -0.0000456, upper bound: 0.0000480
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 3, lower bound: -0.0000441, upper bound: 0.0000493
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 3, lower bound: -0.0000476, upper bound: 0.0000460
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 3, lower bound: -0.0000456, upper bound: 0.0000471

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0001625, 0.0001660
1: -0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000280, 0.0000298
2: 0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002036, 0.0002088
3: 1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000541, 0.0000515
4: -0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000319, 0.0000309
5: 0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001240, 0.0001268
6: -0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000078, 0.0000070
7: -0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0003006, 0.0003055
8: -0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0003302, 0.0003183
9: -0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0001484, 0.0001547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000476, upper bound: 0.0000423
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000475, upper bound: 0.0000424
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0001623, 0.0001656
1: -0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000280, 0.0000297
2: 0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002037, 0.0002082
3: 1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000532, 0.0000523
4: -0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000317, 0.0000309
5: 0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001239, 0.0001264
6: -0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000071, 0.0000076
7: -0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0003060, 0.0002994
8: -0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0003290, 0.0003191
9: -0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0001486, 0.0001544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000462, upper bound: 0.0000437
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000463, upper bound: 0.0000435
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0001656, 0.0001623
1: -0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000297, 0.0000280
2: 0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002082, 0.0002037
3: 1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000523, 0.0000532
4: -0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000309, 0.0000317
5: 0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001264, 0.0001239
6: -0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000076, 0.0000071
7: -0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0002994, 0.0003060
8: -0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0003191, 0.0003290
9: -0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0001544, 0.0001486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000435, upper bound: 0.0000463
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000437, upper bound: 0.0000462
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0034737, -0.0028476, -0.0034737, -0.0028476, -0.0001660, 0.0001625
1: -0.0045173, -0.0044073, -0.0045173, -0.0044073, -0.0000298, 0.0000280
2: 0.0101513, 0.0109434, 0.0101513, 0.0109434, -0.0002088, 0.0002036
3: 1.0087373, 1.0089171, 1.0087373, 1.0089171, -0.0000515, 0.0000541
4: -0.0034030, -0.0032814, -0.0034030, -0.0032814, -0.0000309, 0.0000319
5: 0.0012925, 0.0017710, 0.0012925, 0.0017710, -0.0001268, 0.0001240
6: -0.0025228, -0.0024979, -0.0025228, -0.0024979, -0.0000070, 0.0000078
7: -0.0087833, -0.0076686, -0.0087833, -0.0076686, -0.0003055, 0.0003006
8: -0.0043986, -0.0031357, -0.0043986, -0.0031357, -0.0003183, 0.0003302
9: -0.0026368, -0.0020427, -0.0026368, -0.0020427, -0.0001547, 0.0001484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000475
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000423, upper bound: 0.0000476
time: 0.49 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000476, upper bound: 0.0000423
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000475, upper bound: 0.0000424
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000462, upper bound: 0.0000437
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000463, upper bound: 0.0000435
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000435, upper bound: 0.0000463
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000437, upper bound: 0.0000462
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000424, upper bound: 0.0000475
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0000423, upper bound: 0.0000476

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.70 + 24.67 = 27.37 seconds
