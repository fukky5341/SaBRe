## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00199528


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000766, 0.0000766)
1: (-0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028695, 0.0028695)
2: (0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0034436, 0.0034436)
3: (-0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0253993, 0.0253993)
4: (-0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0019318, 0.0019318)
5: (0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019524, 0.0019524)
6: (0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009496, 0.0009496)
7: (-0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0065824, 0.0065824)
8: (0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0052222, 0.0052222)
9: (0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0093926, 0.0093926)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 2.42 = 3.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0024941, upper bound: 0.0024941

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0024070, upper bound: 0.0024601
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0024601, upper bound: 0.0024070
time: 1.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.97 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.97
Output dim: 2, lower bound: -0.0024070, upper bound: 0.0024601
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.97
Output dim: 2, lower bound: -0.0024601, upper bound: 0.0024070

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000750, 0.0000755
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028079, 0.0028268
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033696, 0.0033923
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0248536, 0.0250208
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0019030, 0.0018903
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019233, 0.0019104
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009292, 0.0009355
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0064844, 0.0064410
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0051444, 0.0051100
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0092527, 0.0091908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021684, upper bound: 0.0022512
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022063, upper bound: 0.0022201
time: 1.10 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000755, 0.0000750
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028268, 0.0028079
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033923, 0.0033696
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0250208, 0.0248536
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018903, 0.0019030
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019104, 0.0019233
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009355, 0.0009292
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0064410, 0.0064844
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0051100, 0.0051444
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0091908, 0.0092527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022202, upper bound: 0.0022063
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022512, upper bound: 0.0021684
time: 1.09 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 2, lower bound: -0.0021684, upper bound: 0.0022512
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 2, lower bound: -0.0022063, upper bound: 0.0022201
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 2, lower bound: -0.0022202, upper bound: 0.0022063
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 2, lower bound: -0.0022512, upper bound: 0.0021684

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000732, 0.0000769
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0027422, 0.0028785
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0032907, 0.0034543
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0242717, 0.0254781
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0019378, 0.0018460
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019585, 0.0018657
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009075, 0.0009526
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0066029, 0.0062902
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0052384, 0.0049904
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0094218, 0.0089756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020521, upper bound: 0.0021040
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020260, upper bound: 0.0021403
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000750, 0.0000737
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028079, 0.0027610
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033696, 0.0033134
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0248536, 0.0244389
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018587, 0.0018903
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018786, 0.0019104
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009292, 0.0009137
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0063336, 0.0064410
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0050247, 0.0051100
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0090375, 0.0091908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020883, upper bound: 0.0020718
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020648, upper bound: 0.0021095
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000737, 0.0000765
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0027610, 0.0028636
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033134, 0.0034365
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0244389, 0.0253469
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0019278, 0.0018587
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019484, 0.0018786
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009137, 0.0009477
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0065689, 0.0063336
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0052114, 0.0050247
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0093733, 0.0090375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021095, upper bound: 0.0020648
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020718, upper bound: 0.0020883
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000755, 0.0000732
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028268, 0.0027422
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033923, 0.0032907
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0250208, 0.0242717
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018460, 0.0019030
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018657, 0.0019233
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009355, 0.0009075
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0062902, 0.0064844
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0049904, 0.0051444
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0089756, 0.0092527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021403, upper bound: 0.0020260
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021040, upper bound: 0.0020521
time: 1.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 2, lower bound: -0.0020521, upper bound: 0.0021040
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 2, lower bound: -0.0020260, upper bound: 0.0021403
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 2, lower bound: -0.0020883, upper bound: 0.0020718
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 2, lower bound: -0.0020648, upper bound: 0.0021095
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 2, lower bound: -0.0021095, upper bound: 0.0020648
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 2, lower bound: -0.0020718, upper bound: 0.0020883
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 2, lower bound: -0.0021403, upper bound: 0.0020260
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 2, lower bound: -0.0021040, upper bound: 0.0020521

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000664, 0.0000690
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0024857, 0.0025851
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0029830, 0.0031022
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0220018, 0.0228814
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017403, 0.0016734
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017588, 0.0016912
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008226, 0.0008555
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0059299, 0.0057020
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0047045, 0.0045237
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0084615, 0.0081363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019397, upper bound: 0.0019745
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019454, upper bound: 0.0019712
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000654, 0.0000700
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0024488, 0.0026202
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0029386, 0.0031444
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0216749, 0.0231923
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017639, 0.0016485
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017827, 0.0016661
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008104, 0.0008671
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0060105, 0.0056172
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0047684, 0.0044564
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0085765, 0.0080154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019155, upper bound: 0.0020069
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019199, upper bound: 0.0020021
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000682, 0.0000659
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0025546, 0.0024677
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0030656, 0.0029613
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0226114, 0.0218421
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0016612, 0.0017197
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0016790, 0.0017381
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008454, 0.0008166
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0056606, 0.0058599
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0044908, 0.0046490
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0080772, 0.0083617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019617, upper bound: 0.0019593
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019647, upper bound: 0.0019524
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000672, 0.0000670
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0025176, 0.0025077
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0030213, 0.0030093
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0222845, 0.0221961
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0016881, 0.0016949
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017062, 0.0017130
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008332, 0.0008299
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0057523, 0.0057752
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0045636, 0.0045818
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0082081, 0.0082408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019409, upper bound: 0.0019945
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019426, upper bound: 0.0019847
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000670, 0.0000686
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0025077, 0.0025703
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0030093, 0.0030844
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0221961, 0.0227501
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017303, 0.0016881
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017488, 0.0017062
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008299, 0.0008506
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0058959, 0.0057523
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0046775, 0.0045636
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0084130, 0.0082081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019847, upper bound: 0.0019426
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019945, upper bound: 0.0019409
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000659, 0.0000695
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0024677, 0.0026029
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0029613, 0.0031236
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0218421, 0.0230392
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017523, 0.0016612
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017710, 0.0016790
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008166, 0.0008614
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0059708, 0.0056606
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0047370, 0.0044908
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0085199, 0.0080772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019523, upper bound: 0.0019647
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019593, upper bound: 0.0019617
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000688, 0.0000654
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0025765, 0.0024488
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0030919, 0.0029386
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0228057, 0.0216749
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0016485, 0.0017345
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0016661, 0.0017530
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008527, 0.0008104
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0056172, 0.0059103
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0044564, 0.0046889
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0080154, 0.0084335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020021, upper bound: 0.0019199
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020068, upper bound: 0.0019155
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000677, 0.0000664
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0025365, 0.0024857
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0030440, 0.0029830
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0224517, 0.0220018
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0016734, 0.0017076
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0016912, 0.0017258
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008394, 0.0008226
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0057020, 0.0058186
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0045237, 0.0046162
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0081363, 0.0083026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019712, upper bound: 0.0019454
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019745, upper bound: 0.0019397
time: 1.09 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019397, upper bound: 0.0019745
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019454, upper bound: 0.0019712
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019155, upper bound: 0.0020069
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019199, upper bound: 0.0020021
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019617, upper bound: 0.0019593
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019647, upper bound: 0.0019524
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019409, upper bound: 0.0019945
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019426, upper bound: 0.0019847
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019847, upper bound: 0.0019426
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019945, upper bound: 0.0019409
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019523, upper bound: 0.0019647
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019593, upper bound: 0.0019617
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0020021, upper bound: 0.0019199
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0020068, upper bound: 0.0019155
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019712, upper bound: 0.0019454
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 2, lower bound: -0.0019745, upper bound: 0.0019397

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000645, 0.0000694
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0024161, 0.0025998
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0028994, 0.0031199
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0213852, 0.0230119
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017502, 0.0016265
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017689, 0.0016438
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007996, 0.0008604
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0059637, 0.0055422
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0047313, 0.0043969
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0085098, 0.0079082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018471, upper bound: 0.0019338
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018471, upper bound: 0.0019338
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000654, 0.0000691
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0024488, 0.0025875
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0029386, 0.0031051
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0216749, 0.0229027
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017419, 0.0016485
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017605, 0.0016661
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008104, 0.0008563
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0059354, 0.0056172
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0047089, 0.0044564
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0084694, 0.0080154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018520, upper bound: 0.0019289
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018520, upper bound: 0.0019289
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000679, 0.0000648
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0025442, 0.0024265
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0030531, 0.0029119
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0225194, 0.0214777
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0016335, 0.0017127
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0016509, 0.0017310
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008420, 0.0008030
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0055661, 0.0058361
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0044159, 0.0046301
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0079424, 0.0083277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019289, upper bound: 0.0018520
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019289, upper bound: 0.0018520
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000688, 0.0000645
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0025765, 0.0024161
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0030919, 0.0028994
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0228057, 0.0213853
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0016265, 0.0017345
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0016438, 0.0017530
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008527, 0.0007996
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0055422, 0.0059103
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0043969, 0.0046889
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0079082, 0.0084335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019337, upper bound: 0.0018471
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019337, upper bound: 0.0018471
time: 1.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.93 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.93
Output dim: 2, lower bound: -0.0018471, upper bound: 0.0019338
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.93
Output dim: 2, lower bound: -0.0018471, upper bound: 0.0019338
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.93
Output dim: 2, lower bound: -0.0018520, upper bound: 0.0019289
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.93
Output dim: 2, lower bound: -0.0018520, upper bound: 0.0019289
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.93
Output dim: 2, lower bound: -0.0019289, upper bound: 0.0018520
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.93
Output dim: 2, lower bound: -0.0019289, upper bound: 0.0018520
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.93
Output dim: 2, lower bound: -0.0019337, upper bound: 0.0018471
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.93
Output dim: 2, lower bound: -0.0019337, upper bound: 0.0018471

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.85 + 81.93 = 85.77 seconds
