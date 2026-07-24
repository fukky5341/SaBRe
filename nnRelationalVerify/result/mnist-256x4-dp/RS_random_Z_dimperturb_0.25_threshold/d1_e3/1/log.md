## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00018112


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000074, 0.0000074)
1: (-0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002773, 0.0002773)
2: (0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0003328, 0.0003328)
3: (0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0024546, 0.0024546)
4: (-0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001867, 0.0001867)
5: (0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001887, 0.0001887)
6: (0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000918, 0.0000918)
7: (-0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0006361, 0.0006361)
8: (0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0005047, 0.0005047)
9: (0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0009077, 0.0009077)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 1.35 = 2.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0002364, upper bound: 0.0002364

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002250, upper bound: 0.0002275
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002275, upper bound: 0.0002251
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.05 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.05
Output dim: 2, lower bound: -0.0002250, upper bound: 0.0002275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.05
Output dim: 2, lower bound: -0.0002275, upper bound: 0.0002251

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000069, 0.0000069
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002567, 0.0002579
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0003080, 0.0003094
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0022720, 0.0022823
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001736, 0.0001728
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001754, 0.0001746
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000849, 0.0000853
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005915, 0.0005888
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0004693, 0.0004671
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0008440, 0.0008402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002020, upper bound: 0.0002031
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002020, upper bound: 0.0002031
time: 0.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000069, 0.0000069
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002579, 0.0002567
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0003094, 0.0003080
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0022823, 0.0022720
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001728, 0.0001736
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001746, 0.0001754
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000853, 0.0000849
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005888, 0.0005915
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0004671, 0.0004693
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0008402, 0.0008440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002128, upper bound: 0.0002188
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0002215, upper bound: 0.0002117
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 2, lower bound: -0.0002020, upper bound: 0.0002031
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 2, lower bound: -0.0002020, upper bound: 0.0002031
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 2, lower bound: -0.0002128, upper bound: 0.0002188
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 2, lower bound: -0.0002215, upper bound: 0.0002117

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000068, 0.0000069
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002557, 0.0002573
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0003069, 0.0003087
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0022634, 0.0022773
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001732, 0.0001721
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001750, 0.0001740
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000846, 0.0000851
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005902, 0.0005866
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0004682, 0.0004654
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0008421, 0.0008370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001964
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001952, upper bound: 0.0001897
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000068, 0.0000069
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002561, 0.0002579
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0003074, 0.0003094
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0022670, 0.0022823
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001736, 0.0001724
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001754, 0.0001743
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000848, 0.0000853
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005915, 0.0005875
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0004693, 0.0004661
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0008440, 0.0008383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001964
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001952, upper bound: 0.0001897
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000059, 0.0000060
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002202, 0.0002249
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002642, 0.0002699
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0019487, 0.0019910
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001514, 0.0001482
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001530, 0.0001498
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000729, 0.0000744
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005160, 0.0005050
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0004094, 0.0004007
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0007363, 0.0007206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001952
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001952
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000060, 0.0000058
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002262, 0.0002190
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002714, 0.0002628
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0020020, 0.0019384
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001474, 0.0001523
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001490, 0.0001539
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000749, 0.0000725
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005024, 0.0005188
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0003985, 0.0004116
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0007168, 0.0007403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001964, upper bound: 0.0001883
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001964, upper bound: 0.0001883
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001964
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 2, lower bound: -0.0001952, upper bound: 0.0001897
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 2, lower bound: -0.0001883, upper bound: 0.0001964
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 2, lower bound: -0.0001952, upper bound: 0.0001897
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001952
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 2, lower bound: -0.0001897, upper bound: 0.0001952
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 2, lower bound: -0.0001964, upper bound: 0.0001883
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 2, lower bound: -0.0001964, upper bound: 0.0001883

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000058, 0.0000060
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002179, 0.0002255
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002615, 0.0002706
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0019289, 0.0019961
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001518, 0.0001467
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001534, 0.0001483
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000721, 0.0000746
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005173, 0.0004999
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0004104, 0.0003966
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0007382, 0.0007133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001483, upper bound: 0.0001505
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001483, upper bound: 0.0001505
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000060, 0.0000059
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002235, 0.0002195
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002682, 0.0002634
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0019779, 0.0019428
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001478, 0.0001504
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001493, 0.0001520
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000740, 0.0000726
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005035, 0.0005126
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0003995, 0.0004067
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0007185, 0.0007314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001567, upper bound: 0.0001732
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001777, upper bound: 0.0001558
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000058, 0.0000060
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002183, 0.0002262
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002620, 0.0002714
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0019325, 0.0020020
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001523, 0.0001470
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001539, 0.0001486
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000723, 0.0000749
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005188, 0.0005008
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0004116, 0.0003973
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0007403, 0.0007147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001553, upper bound: 0.0001784
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001719, upper bound: 0.0001570
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000060, 0.0000059
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002243, 0.0002202
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002691, 0.0002642
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0019851, 0.0019487
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001482, 0.0001510
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001498, 0.0001526
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000742, 0.0000729
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005050, 0.0005145
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0004007, 0.0004082
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0007206, 0.0007341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001492, upper bound: 0.0001494
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001492, upper bound: 0.0001494
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000058, 0.0000060
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002189, 0.0002243
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002627, 0.0002691
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0019375, 0.0019851
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001510, 0.0001474
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001526, 0.0001489
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000724, 0.0000742
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005145, 0.0005021
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0004082, 0.0003984
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0007341, 0.0007165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001817, upper bound: 0.0001807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001731, upper bound: 0.0001872
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000059, 0.0000060
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002195, 0.0002249
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002634, 0.0002699
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0019428, 0.0019910
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001514, 0.0001478
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001530, 0.0001493
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000726, 0.0000744
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005160, 0.0005035
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0004094, 0.0003995
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0007363, 0.0007185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001817, upper bound: 0.0001807
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001731, upper bound: 0.0001872
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000060, 0.0000058
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002247, 0.0002183
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002696, 0.0002620
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0019888, 0.0019325
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001470, 0.0001513
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001486, 0.0001529
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000744, 0.0000723
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005008, 0.0005154
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0003973, 0.0004089
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0007147, 0.0007355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001570, upper bound: 0.0001719
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001784, upper bound: 0.0001553
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000060, 0.0000058
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0002255, 0.0002190
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002706, 0.0002628
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0019961, 0.0019384
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001474, 0.0001518
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001490, 0.0001534
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000746, 0.0000725
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0005024, 0.0005173
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0003985, 0.0004104
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0007168, 0.0007382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001884, upper bound: 0.0001724
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0001813, upper bound: 0.0001804
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001483, upper bound: 0.0001505
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001483, upper bound: 0.0001505
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001567, upper bound: 0.0001732
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001777, upper bound: 0.0001558
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001553, upper bound: 0.0001784
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001719, upper bound: 0.0001570
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001492, upper bound: 0.0001494
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001492, upper bound: 0.0001494
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001817, upper bound: 0.0001807
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001731, upper bound: 0.0001872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001817, upper bound: 0.0001807
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001731, upper bound: 0.0001872
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001570, upper bound: 0.0001719
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001784, upper bound: 0.0001553
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001884, upper bound: 0.0001724
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 2, lower bound: -0.0001813, upper bound: 0.0001804

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000049, 0.0000050
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0001843, 0.0001865
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002212, 0.0002239
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0016313, 0.0016511
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001256, 0.0001241
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001269, 0.0001254
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000610, 0.0000617
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0004279, 0.0004228
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0003395, 0.0003354
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0006106, 0.0006032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001475, upper bound: 0.0001628
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001651, upper bound: 0.0001459
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000048, 0.0000050
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0001812, 0.0001886
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002174, 0.0002263
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0016034, 0.0016692
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001270, 0.0001220
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001283, 0.0001233
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000600, 0.0000624
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0004326, 0.0004155
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0003432, 0.0003297
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0006173, 0.0005930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001434, upper bound: 0.0001694
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001564, upper bound: 0.0001483
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000049, 0.0000050
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0001849, 0.0001875
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002219, 0.0002250
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0016366, 0.0016597
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001262, 0.0001245
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001276, 0.0001258
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000612, 0.0000621
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0004301, 0.0004241
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0003412, 0.0003365
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0006138, 0.0006052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001415, upper bound: 0.0001332
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001415, upper bound: 0.0001332
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000049, 0.0000051
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0001818, 0.0001897
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002181, 0.0002276
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0016088, 0.0016788
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001277, 0.0001224
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001290, 0.0001237
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000602, 0.0000628
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0004351, 0.0004169
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0003452, 0.0003308
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0006208, 0.0005949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001334, upper bound: 0.0001412
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001334, upper bound: 0.0001412
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000051, 0.0000048
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0001902, 0.0001816
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002282, 0.0002179
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0016831, 0.0016071
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001222, 0.0001280
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001235, 0.0001294
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000629, 0.0000601
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0004165, 0.0004362
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0003304, 0.0003461
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0005943, 0.0006224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001424, upper bound: 0.0001320
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001424, upper bound: 0.0001320
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041016, -0.0040838, -0.0041016, -0.0040838, -0.0000050, 0.0000049
1: -0.0062598, -0.0055955, -0.0062598, -0.0055955, -0.0001878, 0.0001846
2: 0.9689516, 0.9697486, 0.9689516, 0.9697486, -0.0002253, 0.0002215
3: 0.0172972, 0.0231766, 0.0172972, 0.0231766, -0.0016621, 0.0016338
4: -0.0024557, -0.0020086, -0.0024557, -0.0020086, -0.0001243, 0.0001264
5: 0.0147884, 0.0152403, 0.0147884, 0.0152403, -0.0001256, 0.0001278
6: 0.0044939, 0.0047137, 0.0044939, 0.0047137, -0.0000621, 0.0000611
7: -0.0137847, -0.0122610, -0.0137847, -0.0122610, -0.0004234, 0.0004308
8: 0.0057930, 0.0070019, 0.0057930, 0.0070019, -0.0003359, 0.0003417
9: 0.0081439, 0.0103181, 0.0081439, 0.0103181, -0.0006042, 0.0006146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 79

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001349, upper bound: 0.0001404
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0001349, upper bound: 0.0001404
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.43 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001475, upper bound: 0.0001628
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001651, upper bound: 0.0001459
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001434, upper bound: 0.0001694
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001564, upper bound: 0.0001483
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001415, upper bound: 0.0001332
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001415, upper bound: 0.0001332
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001334, upper bound: 0.0001412
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001334, upper bound: 0.0001412
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001424, upper bound: 0.0001320
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001424, upper bound: 0.0001320
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001349, upper bound: 0.0001404
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 2, lower bound: -0.0001349, upper bound: 0.0001404

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.69 + 57.29 = 59.98 seconds
