## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00018056


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001558, 0.0001558)
1: (0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0003019, 0.0003019)
2: (0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0024347, 0.0024347)
3: (-0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002175, 0.0002175)
4: (0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0010551, 0.0010551)
5: (-0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001575, 0.0001575)
6: (0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002889, 0.0002889)
7: (-0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0019099, 0.0019099)
8: (0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005983, 0.0005983)
9: (-0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011942, 0.0011942)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 1.41 = 2.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0002194, upper bound: 0.0002194

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002172, upper bound: 0.0002115
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002115, upper bound: 0.0002173
time: 0.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.19 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 6, lower bound: -0.0002172, upper bound: 0.0002115
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 6, lower bound: -0.0002115, upper bound: 0.0002173

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001567, 0.0001561
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0003035, 0.0003023
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0024387, 0.0024478
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002186, 0.0002178
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0010607, 0.0010568
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001578, 0.0001583
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002904, 0.0002893
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0019201, 0.0019129
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0006016, 0.0005993
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011961, 0.0012006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002132, upper bound: 0.0002001
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002044, upper bound: 0.0002074
time: 0.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001561, 0.0001567
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0003023, 0.0003035
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0024478, 0.0024387
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002178, 0.0002186
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0010568, 0.0010607
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001583, 0.0001578
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002893, 0.0002904
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0019129, 0.0019201
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005993, 0.0006016
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0012006, 0.0011961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 237

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002102, upper bound: 0.0002079
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002039, upper bound: 0.0002160
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 6, lower bound: -0.0002132, upper bound: 0.0002001
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 6, lower bound: -0.0002044, upper bound: 0.0002074
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 6, lower bound: -0.0002102, upper bound: 0.0002079
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 6, lower bound: -0.0002039, upper bound: 0.0002160

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001483, 0.0001456
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002872, 0.0002820
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0022742, 0.0023169
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002069, 0.0002031
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0010040, 0.0009855
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001471, 0.0001499
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002749, 0.0002698
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0018174, 0.0017839
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005694, 0.0005589
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011155, 0.0011364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002077, upper bound: 0.0001932
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002045, upper bound: 0.0001948
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001462, 0.0001478
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002831, 0.0002863
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0023089, 0.0022834
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002039, 0.0002062
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009895, 0.0010005
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001494, 0.0001477
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002709, 0.0002739
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017911, 0.0018111
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005611, 0.0005674
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011325, 0.0011200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002031, upper bound: 0.0001990
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001995, upper bound: 0.0002061
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001516, 0.0001505
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002937, 0.0002915
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0023512, 0.0023690
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002116, 0.0002100
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0010266, 0.0010189
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001521, 0.0001532
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002811, 0.0002790
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0018583, 0.0018443
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005822, 0.0005778
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011532, 0.0011620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002014, upper bound: 0.0001891
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001938, upper bound: 0.0001986
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001499, 0.0001522
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002904, 0.0002949
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0023785, 0.0023420
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002092, 0.0002124
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0010149, 0.0010307
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001539, 0.0001515
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002779, 0.0002822
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0018371, 0.0018658
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005756, 0.0005845
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011666, 0.0011487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001990, upper bound: 0.0002079
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001976, upper bound: 0.0002106
time: 0.60 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 6, lower bound: -0.0002077, upper bound: 0.0001932
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 6, lower bound: -0.0002045, upper bound: 0.0001948
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 6, lower bound: -0.0002031, upper bound: 0.0001990
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 6, lower bound: -0.0001995, upper bound: 0.0002061
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 6, lower bound: -0.0002014, upper bound: 0.0001891
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 6, lower bound: -0.0001938, upper bound: 0.0001986
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 6, lower bound: -0.0001990, upper bound: 0.0002079
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 6, lower bound: -0.0001976, upper bound: 0.0002106

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001476, 0.0001443
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002858, 0.0002795
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0022545, 0.0023056
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002059, 0.0002014
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009991, 0.0009770
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001458, 0.0001491
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002735, 0.0002675
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0018086, 0.0017685
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005666, 0.0005540
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011058, 0.0011309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001995, upper bound: 0.0001771
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001874, upper bound: 0.0001831
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001483, 0.0001448
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002872, 0.0002806
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0022629, 0.0023169
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002069, 0.0002021
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0010040, 0.0009806
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001464, 0.0001499
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002749, 0.0002685
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0018174, 0.0017751
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005694, 0.0005561
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011099, 0.0011364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002035, upper bound: 0.0001914
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001965, upper bound: 0.0001937
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001427, 0.0001426
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002765, 0.0002762
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0022281, 0.0022299
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001992, 0.0001990
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009663, 0.0009655
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001441, 0.0001443
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002646, 0.0002643
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017492, 0.0017477
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005480, 0.0005476
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010928, 0.0010938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001927, upper bound: 0.0001889
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001942, upper bound: 0.0001889
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001410, 0.0001445
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002731, 0.0002799
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0022580, 0.0022026
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001967, 0.0002017
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009545, 0.0009785
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001461, 0.0001425
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002613, 0.0002679
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017277, 0.0017712
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005413, 0.0005549
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011075, 0.0010803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001892, upper bound: 0.0001955
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001906, upper bound: 0.0001954
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001489, 0.0001439
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002883, 0.0002786
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0022475, 0.0023257
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002077, 0.0002007
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0010078, 0.0009739
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001454, 0.0001504
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002759, 0.0002666
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0018243, 0.0017630
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005715, 0.0005523
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011024, 0.0011407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001971, upper bound: 0.0001787
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001908, upper bound: 0.0001847
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001450, 0.0001475
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002808, 0.0002858
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0023050, 0.0022653
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002023, 0.0002059
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009816, 0.0009988
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001491, 0.0001465
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002688, 0.0002735
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017769, 0.0018080
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005567, 0.0005664
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011306, 0.0011111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001898, upper bound: 0.0001906
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001795, upper bound: 0.0001934
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001492, 0.0001509
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002890, 0.0002923
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0023580, 0.0023308
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002082, 0.0002106
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0010100, 0.0010218
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001525, 0.0001508
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002765, 0.0002798
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0018283, 0.0018497
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005728, 0.0005795
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011566, 0.0011432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001943, upper bound: 0.0001963
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001914, upper bound: 0.0002035
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001499, 0.0001515
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002904, 0.0002935
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0023673, 0.0023420
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002092, 0.0002114
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0010149, 0.0010258
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001531, 0.0001515
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002779, 0.0002809
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0018371, 0.0018570
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005756, 0.0005818
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011611, 0.0011487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001837, upper bound: 0.0001893
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001801, upper bound: 0.0001947
time: 0.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001995, upper bound: 0.0001771
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001874, upper bound: 0.0001831
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0002035, upper bound: 0.0001914
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001965, upper bound: 0.0001937
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001927, upper bound: 0.0001889
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001942, upper bound: 0.0001889
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001892, upper bound: 0.0001955
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001906, upper bound: 0.0001954
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001971, upper bound: 0.0001787
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001908, upper bound: 0.0001847
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001898, upper bound: 0.0001906
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001795, upper bound: 0.0001934
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001943, upper bound: 0.0001963
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001914, upper bound: 0.0002035
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001837, upper bound: 0.0001893
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 6, lower bound: -0.0001801, upper bound: 0.0001947

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001361, 0.0001279
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002636, 0.0002476
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019974, 0.0021262
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001899, 0.0001784
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009214, 0.0008656
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001292, 0.0001375
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002523, 0.0002370
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016679, 0.0015668
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005225, 0.0004909
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009797, 0.0010429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001853, upper bound: 0.0001654
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001853, upper bound: 0.0001654
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001311, 0.0001323
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002540, 0.0002563
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0020672, 0.0020485
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001830, 0.0001846
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008877, 0.0008958
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001337, 0.0001325
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002430, 0.0002453
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016069, 0.0016216
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005034, 0.0005080
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010139, 0.0010048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001865, upper bound: 0.0001800
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001761, upper bound: 0.0001820
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001451, 0.0001396
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002810, 0.0002704
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021808, 0.0022668
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002025, 0.0001948
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009823, 0.0009450
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001411, 0.0001466
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002689, 0.0002587
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017781, 0.0017107
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005571, 0.0005359
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010697, 0.0011118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001955, upper bound: 0.0001746
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001818, upper bound: 0.0001802
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001431, 0.0001413
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002772, 0.0002737
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0022078, 0.0022361
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001997, 0.0001972
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009690, 0.0009567
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001428, 0.0001446
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002653, 0.0002619
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017540, 0.0017318
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005495, 0.0005426
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010829, 0.0010968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001880, upper bound: 0.0001785
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001737, upper bound: 0.0001826
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001379, 0.0001373
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002671, 0.0002659
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021443, 0.0021543
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001924, 0.0001915
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009336, 0.0009292
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001387, 0.0001394
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002556, 0.0002544
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016899, 0.0016820
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005294, 0.0005270
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010518, 0.0010567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001834, upper bound: 0.0001708
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001751, upper bound: 0.0001791
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001427, 0.0001378
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002765, 0.0002669
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021525, 0.0022299
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001992, 0.0001922
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009663, 0.0009328
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001392, 0.0001443
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002646, 0.0002554
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017492, 0.0016884
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005480, 0.0005290
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010558, 0.0010938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001845, upper bound: 0.0001708
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001754, upper bound: 0.0001789
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001361, 0.0001388
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002637, 0.0002689
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021687, 0.0021270
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001900, 0.0001937
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009217, 0.0009398
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001403, 0.0001376
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002523, 0.0002573
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016684, 0.0017012
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005227, 0.0005330
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010637, 0.0010433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001821, upper bound: 0.0001857
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001818, upper bound: 0.0001872
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001410, 0.0001397
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002731, 0.0002706
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021824, 0.0022026
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001967, 0.0001949
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009545, 0.0009457
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001412, 0.0001425
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002613, 0.0002589
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017277, 0.0017119
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005413, 0.0005363
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010704, 0.0010803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001823, upper bound: 0.0001857
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001822, upper bound: 0.0001872
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001408, 0.0001336
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002727, 0.0002587
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0020870, 0.0021997
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001965, 0.0001864
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009532, 0.0009044
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001350, 0.0001423
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002610, 0.0002476
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017255, 0.0016370
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005406, 0.0005129
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010236, 0.0010789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001861, upper bound: 0.0001603
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001815, upper bound: 0.0001687
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001386, 0.0001357
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002684, 0.0002629
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021205, 0.0021651
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001934, 0.0001894
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009382, 0.0009189
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001372, 0.0001401
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002569, 0.0002516
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016984, 0.0016634
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005321, 0.0005211
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010401, 0.0010620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001800, upper bound: 0.0001773
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001799, upper bound: 0.0001777
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001371, 0.0001373
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002656, 0.0002659
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021444, 0.0021424
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001913, 0.0001915
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009284, 0.0009293
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001387, 0.0001386
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002542, 0.0002544
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016805, 0.0016821
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005265, 0.0005270
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010518, 0.0010508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001803, upper bound: 0.0001708
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001725, upper bound: 0.0001809
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001347, 0.0001392
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002609, 0.0002697
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021750, 0.0021048
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001880, 0.0001943
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009121, 0.0009425
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001407, 0.0001362
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002497, 0.0002581
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016510, 0.0017061
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005173, 0.0005345
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010668, 0.0010324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001688, upper bound: 0.0001820
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001688, upper bound: 0.0001820
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001418, 0.0001413
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002747, 0.0002737
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0022080, 0.0022155
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001979, 0.0001972
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009600, 0.0009568
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001428, 0.0001433
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002628, 0.0002620
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017378, 0.0017320
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005445, 0.0005426
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010830, 0.0010867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001822, upper bound: 0.0001849
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001822, upper bound: 0.0001847
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001396, 0.0001437
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002704, 0.0002784
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0022452, 0.0021808
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001948, 0.0002005
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009450, 0.0009729
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001452, 0.0001411
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002587, 0.0002664
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017107, 0.0017612
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005359, 0.0005518
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011012, 0.0010697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001793, upper bound: 0.0001917
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001793, upper bound: 0.0001917
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001470, 0.0001449
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002848, 0.0002806
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0022636, 0.0022972
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0002052, 0.0002022
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009955, 0.0009809
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001464, 0.0001486
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002725, 0.0002686
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0018019, 0.0017756
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005645, 0.0005563
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011103, 0.0011267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001734, upper bound: 0.0001701
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001649, upper bound: 0.0001802
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001433, 0.0001487
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002775, 0.0002880
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0023230, 0.0022383
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001999, 0.0002075
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009700, 0.0010066
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001503, 0.0001448
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002656, 0.0002756
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017558, 0.0018222
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005501, 0.0005709
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0011394, 0.0010979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001605, upper bound: 0.0001715
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001605, upper bound: 0.0001715
time: 0.63 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001853, upper bound: 0.0001654
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001853, upper bound: 0.0001654
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001865, upper bound: 0.0001800
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001761, upper bound: 0.0001820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001955, upper bound: 0.0001746
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001818, upper bound: 0.0001802
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001880, upper bound: 0.0001785
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001737, upper bound: 0.0001826
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001834, upper bound: 0.0001708
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001751, upper bound: 0.0001791
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001845, upper bound: 0.0001708
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001754, upper bound: 0.0001789
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001821, upper bound: 0.0001857
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001818, upper bound: 0.0001872
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001823, upper bound: 0.0001857
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001822, upper bound: 0.0001872
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001861, upper bound: 0.0001603
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001815, upper bound: 0.0001687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001800, upper bound: 0.0001773
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001799, upper bound: 0.0001777
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001803, upper bound: 0.0001708
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001725, upper bound: 0.0001809
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001688, upper bound: 0.0001820
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001688, upper bound: 0.0001820
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001822, upper bound: 0.0001849
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001822, upper bound: 0.0001847
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001793, upper bound: 0.0001917
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001793, upper bound: 0.0001917
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001734, upper bound: 0.0001701
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001649, upper bound: 0.0001802
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001605, upper bound: 0.0001715
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 6, lower bound: -0.0001605, upper bound: 0.0001715

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001317, 0.0001237
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002552, 0.0002395
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019318, 0.0020582
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001838, 0.0001725
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008919, 0.0008371
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001250, 0.0001331
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002442, 0.0002292
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016145, 0.0015154
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005058, 0.0004748
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009475, 0.0010095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001591, upper bound: 0.0001410
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001574, upper bound: 0.0001436
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001361, 0.0001235
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002636, 0.0002392
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019294, 0.0021262
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001899, 0.0001723
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009214, 0.0008361
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001248, 0.0001375
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002523, 0.0002289
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016679, 0.0015134
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005225, 0.0004741
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009463, 0.0010429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001844, upper bound: 0.0001616
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001766, upper bound: 0.0001644
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001271, 0.0001264
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002463, 0.0002448
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019742, 0.0019863
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001774, 0.0001763
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008607, 0.0008555
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001277, 0.0001285
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002357, 0.0002342
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015581, 0.0015486
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004881, 0.0004852
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009683, 0.0009742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001699, upper bound: 0.0001606
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001661, upper bound: 0.0001660
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001252, 0.0001279
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002424, 0.0002476
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019974, 0.0019556
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001747, 0.0001784
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008474, 0.0008656
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001292, 0.0001265
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002320, 0.0002370
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015340, 0.0015668
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004806, 0.0004909
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009797, 0.0009592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001635, upper bound: 0.0001702
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001635, upper bound: 0.0001700
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001327, 0.0001224
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002570, 0.0002372
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019129, 0.0020730
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001851, 0.0001708
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008983, 0.0008289
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001237, 0.0001341
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002459, 0.0002269
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016261, 0.0015005
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005094, 0.0004701
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009382, 0.0010168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001832, upper bound: 0.0001621
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001832, upper bound: 0.0001621
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001279, 0.0001268
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002478, 0.0002457
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019815, 0.0019989
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001785, 0.0001770
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008662, 0.0008586
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001282, 0.0001293
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002372, 0.0002351
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015680, 0.0015543
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004912, 0.0004869
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009719, 0.0009804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001688, upper bound: 0.0001606
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001637, upper bound: 0.0001660
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001310, 0.0001242
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002536, 0.0002405
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019398, 0.0020459
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001827, 0.0001733
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008866, 0.0008406
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001255, 0.0001323
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002427, 0.0002301
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016048, 0.0015216
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005028, 0.0004767
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009515, 0.0010035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001762, upper bound: 0.0001654
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001763, upper bound: 0.0001654
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001260, 0.0001283
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002440, 0.0002485
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0020044, 0.0019682
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001758, 0.0001790
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008529, 0.0008686
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001297, 0.0001273
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002335, 0.0002378
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015439, 0.0015723
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004837, 0.0004926
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009831, 0.0009654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001626, upper bound: 0.0001703
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001627, upper bound: 0.0001702
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001250, 0.0001201
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002421, 0.0002326
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0018764, 0.0019529
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001744, 0.0001676
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008463, 0.0008131
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001214, 0.0001263
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002317, 0.0002226
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015319, 0.0014719
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004799, 0.0004611
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009204, 0.0009579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001727, upper bound: 0.0001538
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001610, upper bound: 0.0001596
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001298, 0.0001206
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002515, 0.0002336
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0018845, 0.0020285
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001812, 0.0001683
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008790, 0.0008166
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001219, 0.0001312
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002407, 0.0002236
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015912, 0.0014783
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004985, 0.0004631
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009244, 0.0009950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001731, upper bound: 0.0001537
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001620, upper bound: 0.0001596
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001354, 0.0001374
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002622, 0.0002662
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021470, 0.0021151
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001889, 0.0001918
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009166, 0.0009304
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001389, 0.0001368
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002509, 0.0002547
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016592, 0.0016842
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005198, 0.0005276
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010531, 0.0010375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001560, upper bound: 0.0001641
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001542, upper bound: 0.0001657
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001361, 0.0001381
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002637, 0.0002674
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021569, 0.0021270
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001900, 0.0001926
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009217, 0.0009347
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001395, 0.0001376
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002523, 0.0002559
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016684, 0.0016919
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005227, 0.0005301
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010579, 0.0010433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001722, upper bound: 0.0001705
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001617, upper bound: 0.0001770
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001402, 0.0001384
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002715, 0.0002680
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021619, 0.0021900
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001956, 0.0001931
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009490, 0.0009368
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001399, 0.0001417
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002598, 0.0002565
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017178, 0.0016958
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005382, 0.0005313
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010604, 0.0010741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001728, upper bound: 0.0001685
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001624, upper bound: 0.0001759
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001410, 0.0001389
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002731, 0.0002691
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021706, 0.0022026
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001967, 0.0001939
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009545, 0.0009406
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001404, 0.0001425
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002613, 0.0002575
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017277, 0.0017026
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005413, 0.0005334
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010646, 0.0010803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001560, upper bound: 0.0001641
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001542, upper bound: 0.0001657
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001278, 0.0001164
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002475, 0.0002254
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0018181, 0.0019959
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001783, 0.0001624
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008649, 0.0007879
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001176, 0.0001291
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002368, 0.0002157
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015656, 0.0014262
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004905, 0.0004468
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0008918, 0.0009790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001740, upper bound: 0.0001498
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001740, upper bound: 0.0001498
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001236, 0.0001209
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002394, 0.0002343
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0018895, 0.0019308
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001725, 0.0001688
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008367, 0.0008188
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001222, 0.0001249
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002291, 0.0002242
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015146, 0.0014822
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004745, 0.0004644
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009268, 0.0009470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001682, upper bound: 0.0001586
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001682, upper bound: 0.0001581
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001199, 0.0001247
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002323, 0.0002415
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019478, 0.0018735
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001673, 0.0001740
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008119, 0.0008440
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001260, 0.0001212
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002223, 0.0002311
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0014696, 0.0015279
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004604, 0.0004787
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009554, 0.0009189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001611, upper bound: 0.0001691
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001610, upper bound: 0.0001687
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001299, 0.0001340
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002516, 0.0002595
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0020934, 0.0020292
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001812, 0.0001870
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008793, 0.0009072
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001354, 0.0001313
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002407, 0.0002484
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015917, 0.0016421
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004987, 0.0005145
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010268, 0.0009953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001578, upper bound: 0.0001600
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001524, upper bound: 0.0001729
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001347, 0.0001344
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002609, 0.0002603
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0020994, 0.0021048
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001880, 0.0001875
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009121, 0.0009098
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001358, 0.0001362
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002497, 0.0002491
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016510, 0.0016468
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005173, 0.0005159
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010298, 0.0010324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001579, upper bound: 0.0001600
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001524, upper bound: 0.0001727
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001370, 0.0001361
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002654, 0.0002637
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021267, 0.0021407
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001912, 0.0001899
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009276, 0.0009216
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001376, 0.0001385
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002540, 0.0002523
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016792, 0.0016682
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005261, 0.0005226
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010431, 0.0010500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001721, upper bound: 0.0001661
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001633, upper bound: 0.0001756
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001418, 0.0001365
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002747, 0.0002645
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021332, 0.0022155
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001979, 0.0001905
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009600, 0.0009244
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001380, 0.0001433
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002628, 0.0002531
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017378, 0.0016733
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005445, 0.0005242
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010463, 0.0010867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001578, upper bound: 0.0001583
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001572, upper bound: 0.0001610
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001348, 0.0001380
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002611, 0.0002673
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021561, 0.0021060
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001881, 0.0001926
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009126, 0.0009343
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001395, 0.0001362
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002499, 0.0002558
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016520, 0.0016913
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005176, 0.0005299
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010575, 0.0010330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001685, upper bound: 0.0001702
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001621, upper bound: 0.0001832
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001396, 0.0001389
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002704, 0.0002691
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0021704, 0.0021808
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001948, 0.0001938
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0009450, 0.0009405
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001404, 0.0001411
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002587, 0.0002575
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0017107, 0.0017025
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005359, 0.0005334
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0010645, 0.0010697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001686, upper bound: 0.0001702
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001621, upper bound: 0.0001832
time: 0.62 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001591, upper bound: 0.0001410
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001574, upper bound: 0.0001436
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001844, upper bound: 0.0001616
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001766, upper bound: 0.0001644
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001699, upper bound: 0.0001606
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001661, upper bound: 0.0001660
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001635, upper bound: 0.0001702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001635, upper bound: 0.0001700
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001832, upper bound: 0.0001621
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001832, upper bound: 0.0001621
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001688, upper bound: 0.0001606
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001637, upper bound: 0.0001660
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001762, upper bound: 0.0001654
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001763, upper bound: 0.0001654
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001626, upper bound: 0.0001703
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001627, upper bound: 0.0001702
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001727, upper bound: 0.0001538
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001610, upper bound: 0.0001596
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001731, upper bound: 0.0001537
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001620, upper bound: 0.0001596
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001560, upper bound: 0.0001641
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001542, upper bound: 0.0001657
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001722, upper bound: 0.0001705
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001617, upper bound: 0.0001770
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001728, upper bound: 0.0001685
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001624, upper bound: 0.0001759
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001560, upper bound: 0.0001641
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001542, upper bound: 0.0001657
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001740, upper bound: 0.0001498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001740, upper bound: 0.0001498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001682, upper bound: 0.0001586
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001682, upper bound: 0.0001581
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001611, upper bound: 0.0001691
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001610, upper bound: 0.0001687
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001578, upper bound: 0.0001600
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001524, upper bound: 0.0001729
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001579, upper bound: 0.0001600
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001524, upper bound: 0.0001727
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001721, upper bound: 0.0001661
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001633, upper bound: 0.0001756
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001578, upper bound: 0.0001583
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001572, upper bound: 0.0001610
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001685, upper bound: 0.0001702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001621, upper bound: 0.0001832
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001686, upper bound: 0.0001702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 6, lower bound: -0.0001621, upper bound: 0.0001832

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001319, 0.0001171
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002554, 0.0002268
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0018296, 0.0020604
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001840, 0.0001634
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008928, 0.0007928
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001184, 0.0001333
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002444, 0.0002171
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016162, 0.0014352
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005063, 0.0004496
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0008974, 0.0010106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001351
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001567, upper bound: 0.0001363
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001278, 0.0001177
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002476, 0.0002281
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0018395, 0.0019974
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001784, 0.0001643
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008655, 0.0007971
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001190, 0.0001292
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002370, 0.0002182
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015668, 0.0014429
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004909, 0.0004521
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009023, 0.0009797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001351
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001566, upper bound: 0.0001363
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001327, 0.0001177
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002570, 0.0002279
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0018381, 0.0020730
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001851, 0.0001642
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008983, 0.0007965
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001189, 0.0001341
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002459, 0.0002181
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0016261, 0.0014418
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0005094, 0.0004517
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009015, 0.0010168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001351
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001566, upper bound: 0.0001363
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001177, 0.0001258
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002279, 0.0002437
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019660, 0.0018381
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001642, 0.0001756
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0007965, 0.0008520
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001272, 0.0001189
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002181, 0.0002333
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0014418, 0.0015422
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004517, 0.0004832
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009643, 0.0009015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001363, upper bound: 0.0001567
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001351, upper bound: 0.0001583
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068513, 0.0071401, 0.0068513, 0.0071401, -0.0001224, 0.0001266
1: 0.0013524, 0.0019119, 0.0013524, 0.0019119, -0.0002372, 0.0002452
2: 0.0015069, 0.0060196, 0.0015069, 0.0060196, -0.0019776, 0.0019129
3: -0.0030502, -0.0026472, -0.0030502, -0.0026472, -0.0001708, 0.0001766
4: 0.0065371, 0.0084926, 0.0065371, 0.0084926, -0.0008289, 0.0008570
5: -0.0017875, -0.0014956, -0.0017875, -0.0014956, -0.0001279, 0.0001237
6: 0.9929961, 0.9935315, 0.9929961, 0.9935315, -0.0002269, 0.0002346
7: -0.0015496, 0.0019902, -0.0015496, 0.0019902, -0.0015005, 0.0015513
8: 0.0005029, 0.0016119, 0.0005029, 0.0016119, -0.0004701, 0.0004860
9: -0.0105462, -0.0083328, -0.0105462, -0.0083328, -0.0009700, 0.0009382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001363, upper bound: 0.0001567
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001351, upper bound: 0.0001583
time: 0.66 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001351
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 6, lower bound: -0.0001567, upper bound: 0.0001363
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001351
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 6, lower bound: -0.0001566, upper bound: 0.0001363
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 6, lower bound: -0.0001583, upper bound: 0.0001351
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 6, lower bound: -0.0001566, upper bound: 0.0001363
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 6, lower bound: -0.0001363, upper bound: 0.0001567
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 6, lower bound: -0.0001351, upper bound: 0.0001583
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 6, lower bound: -0.0001363, upper bound: 0.0001567
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 6, lower bound: -0.0001351, upper bound: 0.0001583

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.83 + 145.80 = 148.63 seconds
