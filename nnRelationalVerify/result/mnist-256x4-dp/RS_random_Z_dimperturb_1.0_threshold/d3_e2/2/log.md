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
execution time: IAR + RelationalAnalysis = 1.38 + 2.39 = 3.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0024941, upper bound: 0.0024941

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0023098, upper bound: 0.0023098
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0023098, upper bound: 0.0023098
time: 1.19 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 2, lower bound: -0.0023098, upper bound: 0.0023098
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 2, lower bound: -0.0023098, upper bound: 0.0023098

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000765, 0.0000765
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028642, 0.0028652
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0034372, 0.0034384
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0253519, 0.0253612
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0019289, 0.0019282
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019495, 0.0019487
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009479, 0.0009482
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0065726, 0.0065702
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0052144, 0.0052125
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0093785, 0.0093751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022378, upper bound: 0.0022805
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022805, upper bound: 0.0022378
time: 1.18 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000765, 0.0000766
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028652, 0.0028695
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0034384, 0.0034436
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0253611, 0.0253993
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0019318, 0.0019289
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019524, 0.0019495
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009482, 0.0009496
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0065824, 0.0065726
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0052222, 0.0052144
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0093926, 0.0093785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017284, upper bound: 0.0017284
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017284, upper bound: 0.0017284
time: 0.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 2, lower bound: -0.0022378, upper bound: 0.0022805
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 2, lower bound: -0.0022805, upper bound: 0.0022378
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.04
Output dim: 2, lower bound: -0.0017284, upper bound: 0.0017284
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.04
Output dim: 2, lower bound: -0.0017284, upper bound: 0.0017284

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000748, 0.0000754
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028026, 0.0028225
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033632, 0.0033872
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0248066, 0.0249831
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0019001, 0.0018867
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019204, 0.0019068
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009275, 0.0009341
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0064746, 0.0064289
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0051366, 0.0051003
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0092387, 0.0091735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021845, upper bound: 0.0022716
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022289, upper bound: 0.0022247
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000753, 0.0000749
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0028211, 0.0028036
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033855, 0.0033645
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0249706, 0.0248158
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018874, 0.0018992
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019075, 0.0019194
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009336, 0.0009278
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0064312, 0.0064714
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0051022, 0.0051341
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0091769, 0.0092341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022563, upper bound: 0.0022126
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022545, upper bound: 0.0022129
time: 1.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.95 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 2, lower bound: -0.0021845, upper bound: 0.0022716
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 2, lower bound: -0.0022289, upper bound: 0.0022247
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 2, lower bound: -0.0022563, upper bound: 0.0022126
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 2, lower bound: -0.0022545, upper bound: 0.0022129

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000740, 0.0000750
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0027723, 0.0028083
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033269, 0.0033700
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0245388, 0.0248568
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018905, 0.0018663
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0019107, 0.0018862
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009175, 0.0009294
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0064418, 0.0063595
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0051107, 0.0050453
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0091920, 0.0090744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021590, upper bound: 0.0022456
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021590, upper bound: 0.0022473
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000744, 0.0000746
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0027868, 0.0027923
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033443, 0.0033509
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0246671, 0.0247153
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018797, 0.0018761
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018998, 0.0018961
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009223, 0.0009241
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0064052, 0.0063927
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0050816, 0.0050717
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0091397, 0.0091219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021204, upper bound: 0.0020790
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020917, upper bound: 0.0021195
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000739, 0.0000733
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0027663, 0.0027460
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033196, 0.0032954
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0244850, 0.0243061
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018486, 0.0018622
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018684, 0.0018821
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009155, 0.0009088
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0062991, 0.0063455
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0049974, 0.0050342
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0089884, 0.0090545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021826, upper bound: 0.0022084
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022520, upper bound: 0.0021338
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000738, 0.0000734
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0027635, 0.0027486
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0033164, 0.0032984
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0244609, 0.0243286
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018503, 0.0018604
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018701, 0.0018803
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0009146, 0.0009096
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0063050, 0.0063393
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0050021, 0.0050293
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0089967, 0.0090456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016689, upper bound: 0.0016416
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016689, upper bound: 0.0016416
time: 0.85 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 8.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.63
Output dim: 2, lower bound: -0.0021590, upper bound: 0.0022456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.63
Output dim: 2, lower bound: -0.0021590, upper bound: 0.0022473
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.63
Output dim: 2, lower bound: -0.0021204, upper bound: 0.0020790
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.63
Output dim: 2, lower bound: -0.0020917, upper bound: 0.0021195
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.63
Output dim: 2, lower bound: -0.0021826, upper bound: 0.0022084
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.63
Output dim: 2, lower bound: -0.0022520, upper bound: 0.0021338
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.63
Output dim: 2, lower bound: -0.0016689, upper bound: 0.0016416
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.63
Output dim: 2, lower bound: -0.0016689, upper bound: 0.0016416

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000726, 0.0000735
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0027178, 0.0027508
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0032615, 0.0033010
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0240560, 0.0243479
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018518, 0.0018296
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018716, 0.0018491
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008994, 0.0009103
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0063100, 0.0062343
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0050060, 0.0049460
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0090038, 0.0088959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021226, upper bound: 0.0022138
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021273, upper bound: 0.0022105
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000725, 0.0000735
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0027148, 0.0027527
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0032579, 0.0033033
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0240300, 0.0243647
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018531, 0.0018276
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018729, 0.0018471
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008984, 0.0009110
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0063143, 0.0062276
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0050095, 0.0049407
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0090100, 0.0088863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016130, upper bound: 0.0016612
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016130, upper bound: 0.0016612
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000672, 0.0000664
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0025161, 0.0024850
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0030194, 0.0029822
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0222704, 0.0219960
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0016729, 0.0016938
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0016908, 0.0017119
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008327, 0.0008224
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0057004, 0.0057716
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0045225, 0.0045789
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0081341, 0.0082356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015290, upper bound: 0.0015264
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015290, upper bound: 0.0015264
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000662, 0.0000674
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0024796, 0.0025250
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0029756, 0.0030302
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0219478, 0.0223499
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0016998, 0.0016693
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017180, 0.0016871
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008206, 0.0008356
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0057922, 0.0056880
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0045952, 0.0045126
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0082650, 0.0081163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015173, upper bound: 0.0015484
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015173, upper bound: 0.0015484
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000712, 0.0000719
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026658, 0.0026934
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031990, 0.0032322
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0235955, 0.0238401
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018132, 0.0017946
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018325, 0.0018137
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008822, 0.0008913
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0061784, 0.0061150
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0049016, 0.0048513
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0088160, 0.0087256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020680, upper bound: 0.0021005
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020800, upper bound: 0.0020948
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000723, 0.0000706
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0027086, 0.0026456
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0032504, 0.0031748
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0239745, 0.0234166
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017810, 0.0018234
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018000, 0.0018429
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008964, 0.0008755
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0060686, 0.0062132
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048146, 0.0049293
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0086595, 0.0088657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021433, upper bound: 0.0019943
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021081, upper bound: 0.0020238
time: 1.16 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0021226, upper bound: 0.0022138
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0021273, upper bound: 0.0022105
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0016130, upper bound: 0.0016612
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0016130, upper bound: 0.0016612
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0015290, upper bound: 0.0015264
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0015290, upper bound: 0.0015264
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0015173, upper bound: 0.0015484
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0015173, upper bound: 0.0015484
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0020680, upper bound: 0.0021005
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0020800, upper bound: 0.0020948
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0021433, upper bound: 0.0019943
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 2, lower bound: -0.0021081, upper bound: 0.0020238

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000718, 0.0000730
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026885, 0.0027350
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0032264, 0.0032821
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0237970, 0.0242084
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018412, 0.0018099
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018608, 0.0018292
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008897, 0.0009051
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0062738, 0.0061672
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0049774, 0.0048928
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0089523, 0.0088001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015805, upper bound: 0.0016301
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015805, upper bound: 0.0016301
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000720, 0.0000727
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026976, 0.0027215
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0032373, 0.0032659
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0238776, 0.0240889
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018321, 0.0018160
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018517, 0.0018354
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008927, 0.0009006
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0062428, 0.0061881
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0049528, 0.0049093
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0089080, 0.0088299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020160, upper bound: 0.0020990
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020218, upper bound: 0.0020870
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000703, 0.0000713
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026341, 0.0026709
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031610, 0.0032052
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0233151, 0.0236412
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017981, 0.0017733
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018172, 0.0017922
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008717, 0.0008839
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0061268, 0.0060423
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048607, 0.0047937
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0087425, 0.0086219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020284, upper bound: 0.0020916
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020586, upper bound: 0.0020490
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000712, 0.0000711
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026658, 0.0026617
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031990, 0.0031942
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0235955, 0.0235597
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017919, 0.0017946
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018110, 0.0018137
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008822, 0.0008809
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0061057, 0.0061150
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048440, 0.0048513
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0087124, 0.0087256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020377, upper bound: 0.0020857
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020707, upper bound: 0.0020435
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000649, 0.0000623
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0024299, 0.0023314
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0029160, 0.0027978
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0215077, 0.0206359
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015695, 0.0016358
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015862, 0.0016532
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008041, 0.0007715
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0053480, 0.0055739
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0042428, 0.0044221
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0076311, 0.0079535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020771, upper bound: 0.0019278
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020771, upper bound: 0.0019278
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000639, 0.0000632
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0023944, 0.0023683
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0028734, 0.0028421
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0211938, 0.0209629
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015943, 0.0016119
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0016114, 0.0016291
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007924, 0.0007838
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0054327, 0.0054925
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0043101, 0.0043575
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0077520, 0.0078374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020705, upper bound: 0.0019923
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020753, upper bound: 0.0019838
time: 1.19 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0015805, upper bound: 0.0016301
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0015805, upper bound: 0.0016301
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0020160, upper bound: 0.0020990
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0020218, upper bound: 0.0020870
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0020284, upper bound: 0.0020916
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0020586, upper bound: 0.0020490
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0020377, upper bound: 0.0020857
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0020707, upper bound: 0.0020435
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0020771, upper bound: 0.0019278
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0020771, upper bound: 0.0019278
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0020705, upper bound: 0.0019923
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 2, lower bound: -0.0020753, upper bound: 0.0019838

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000712, 0.0000722
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026659, 0.0027019
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031992, 0.0032424
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0235965, 0.0239151
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018189, 0.0017947
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018383, 0.0018138
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008822, 0.0008941
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0061978, 0.0061152
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0049170, 0.0048515
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0088438, 0.0087260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019089, upper bound: 0.0019589
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018756, upper bound: 0.0019872
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000720, 0.0000718
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026976, 0.0026897
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0032373, 0.0032278
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0238776, 0.0238078
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0018107, 0.0018160
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018301, 0.0018354
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008927, 0.0008901
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0061700, 0.0061881
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048950, 0.0049093
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0088041, 0.0088299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019706, upper bound: 0.0020829
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020175, upper bound: 0.0020124
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000699, 0.0000714
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026173, 0.0026733
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031408, 0.0032081
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0231662, 0.0236623
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017997, 0.0017619
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018189, 0.0017807
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008661, 0.0008847
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0061323, 0.0060037
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048651, 0.0047631
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0087503, 0.0085668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019453, upper bound: 0.0020134
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019453, upper bound: 0.0020134
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000703, 0.0000709
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026313, 0.0026541
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031577, 0.0031850
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0232903, 0.0234923
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017867, 0.0017714
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018058, 0.0017903
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008708, 0.0008783
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0060882, 0.0060359
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048301, 0.0047886
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0086874, 0.0086127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0013624
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0013624
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000708, 0.0000711
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026503, 0.0026643
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031805, 0.0031973
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0234585, 0.0235827
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017936, 0.0017842
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018128, 0.0018032
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008771, 0.0008817
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0061117, 0.0060795
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048487, 0.0048232
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0087209, 0.0086749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013721, upper bound: 0.0013754
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013721, upper bound: 0.0013754
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000711, 0.0000706
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026643, 0.0026449
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031973, 0.0031740
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0235826, 0.0234107
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017805, 0.0017936
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017995, 0.0018127
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008817, 0.0008753
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0060671, 0.0061116
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048133, 0.0048487
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0086573, 0.0087208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0013624
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0013624
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000638, 0.0000611
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0023895, 0.0022874
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0028675, 0.0027450
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0211502, 0.0202468
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015399, 0.0016086
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015563, 0.0016258
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007908, 0.0007570
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0052471, 0.0054813
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0041628, 0.0043486
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0074873, 0.0078213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014828, upper bound: 0.0014190
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014828, upper bound: 0.0014190
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000649, 0.0000612
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0024299, 0.0022910
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0029160, 0.0027493
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0215077, 0.0202785
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015423, 0.0016358
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015588, 0.0016532
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008041, 0.0007582
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0052553, 0.0055739
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0041693, 0.0044221
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0074990, 0.0079535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020233, upper bound: 0.0019186
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020680, upper bound: 0.0018923
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000627, 0.0000623
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0023463, 0.0023325
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0028157, 0.0027991
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0207680, 0.0206456
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015702, 0.0015795
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015870, 0.0015964
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007765, 0.0007719
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0053505, 0.0053822
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0042448, 0.0042700
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0076347, 0.0076800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020094, upper bound: 0.0019829
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020615, upper bound: 0.0019615
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000628, 0.0000620
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0023507, 0.0023202
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0028209, 0.0027844
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0208065, 0.0205371
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015620, 0.0015825
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015786, 0.0015993
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007779, 0.0007678
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0053224, 0.0053922
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0042225, 0.0042779
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0075946, 0.0076942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020128, upper bound: 0.0019743
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020663, upper bound: 0.0019543
time: 1.06 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0019089, upper bound: 0.0019589
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0018756, upper bound: 0.0019872
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0019706, upper bound: 0.0020829
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0020175, upper bound: 0.0020124
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0019453, upper bound: 0.0020134
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0019453, upper bound: 0.0020134
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0013624
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0013624
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0013721, upper bound: 0.0013754
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0013721, upper bound: 0.0013754
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0013624
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0013817, upper bound: 0.0013624
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0014828, upper bound: 0.0014190
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0014828, upper bound: 0.0014190
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0020233, upper bound: 0.0019186
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0020680, upper bound: 0.0018923
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0020094, upper bound: 0.0019829
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0020615, upper bound: 0.0019615
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0020128, upper bound: 0.0019743
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 2, lower bound: -0.0020663, upper bound: 0.0019543

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000697, 0.0000708
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026093, 0.0026504
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031313, 0.0031806
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0230961, 0.0234598
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017843, 0.0017566
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0018033, 0.0017753
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008635, 0.0008771
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0060798, 0.0059856
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048234, 0.0047487
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0086754, 0.0085409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018637, upper bound: 0.0019427
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018295, upper bound: 0.0019718
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000707, 0.0000695
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026478, 0.0026008
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031775, 0.0031211
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0234368, 0.0230209
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017509, 0.0017825
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017696, 0.0018015
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008763, 0.0008607
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0059661, 0.0060738
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0047332, 0.0048187
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0085131, 0.0086669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019092, upper bound: 0.0018712
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018781, upper bound: 0.0019035
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000690, 0.0000705
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0025852, 0.0026412
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031024, 0.0031695
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0228826, 0.0233779
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017780, 0.0017404
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017970, 0.0017589
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008555, 0.0008741
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0060586, 0.0059302
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048066, 0.0047048
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0086451, 0.0084620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019116, upper bound: 0.0019818
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019132, upper bound: 0.0019790
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000699, 0.0000705
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0026173, 0.0026413
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0031408, 0.0031696
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0231662, 0.0233787
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0017781, 0.0017619
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0017971, 0.0017807
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0008661, 0.0008741
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0060588, 0.0060037
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0048068, 0.0047631
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0086454, 0.0085668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015468, upper bound: 0.0015950
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015468, upper bound: 0.0015950
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000639, 0.0000606
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0023947, 0.0022707
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0028738, 0.0027250
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0211966, 0.0200991
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015287, 0.0016121
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015450, 0.0016293
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007925, 0.0007515
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0052088, 0.0054933
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0041325, 0.0043581
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0074326, 0.0078385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018937, upper bound: 0.0018125
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019012, upper bound: 0.0018028
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000644, 0.0000602
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0024119, 0.0022554
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0028943, 0.0027066
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0213482, 0.0199631
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015183, 0.0016237
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015345, 0.0016410
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007982, 0.0007464
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0051736, 0.0055326
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0041045, 0.0043893
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0073823, 0.0078945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014751, upper bound: 0.0013900
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014751, upper bound: 0.0013900
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000619, 0.0000619
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0023168, 0.0023181
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0027802, 0.0027818
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0205065, 0.0205183
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015605, 0.0015596
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015772, 0.0015763
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007667, 0.0007671
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0053175, 0.0053144
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0042187, 0.0042162
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0075877, 0.0075833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016558, upper bound: 0.0016481
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016561, upper bound: 0.0016475
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000623, 0.0000615
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0023349, 0.0023029
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0028019, 0.0027636
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0206666, 0.0203841
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015503, 0.0015718
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015669, 0.0015886
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007727, 0.0007621
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0052827, 0.0053559
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0041911, 0.0042491
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0075380, 0.0076425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019432, upper bound: 0.0018581
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019552, upper bound: 0.0018500
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000620, 0.0000615
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0023211, 0.0023047
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0027855, 0.0027657
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0205450, 0.0203993
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015515, 0.0015626
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015680, 0.0015793
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007681, 0.0007627
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0052867, 0.0053244
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0041942, 0.0042241
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0075436, 0.0075975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019002, upper bound: 0.0018753
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0019114, upper bound: 0.0018659
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000625, 0.0000612
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0023403, 0.0022907
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0028084, 0.0027489
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0207143, 0.0202756
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015421, 0.0015754
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015585, 0.0015923
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007745, 0.0007581
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0052546, 0.0053683
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0041688, 0.0042590
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0074979, 0.0076601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020003, upper bound: 0.0018837
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020003, upper bound: 0.0018837
time: 1.16 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0018637, upper bound: 0.0019427
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0018295, upper bound: 0.0019718
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0019092, upper bound: 0.0018712
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0018781, upper bound: 0.0019035
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0019116, upper bound: 0.0019818
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0019132, upper bound: 0.0019790
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0015468, upper bound: 0.0015950
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0015468, upper bound: 0.0015950
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0018937, upper bound: 0.0018125
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0019012, upper bound: 0.0018028
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0014751, upper bound: 0.0013900
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0014751, upper bound: 0.0013900
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0016558, upper bound: 0.0016481
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0016561, upper bound: 0.0016475
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0019432, upper bound: 0.0018581
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0019552, upper bound: 0.0018500
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0019002, upper bound: 0.0018753
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0019114, upper bound: 0.0018659
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0020003, upper bound: 0.0018837
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 2, lower bound: -0.0020003, upper bound: 0.0018837

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000613, 0.0000597
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0022960, 0.0022355
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0027553, 0.0026827
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0203223, 0.0197868
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015049, 0.0015456
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015210, 0.0015621
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007598, 0.0007398
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0051279, 0.0052667
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0040683, 0.0041784
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0073171, 0.0075152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014245, upper bound: 0.0013730
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014245, upper bound: 0.0013730
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041780, -0.0040933, -0.0041780, -0.0040933, -0.0000625, 0.0000600
1: -0.0091216, -0.0059511, -0.0091216, -0.0059511, -0.0023403, 0.0022464
2: 0.9655171, 0.9693220, 0.9655171, 0.9693220, -0.0028084, 0.0026958
3: -0.0080341, 0.0200298, -0.0080341, 0.0200298, -0.0207143, 0.0198836
4: -0.0022164, -0.0000820, -0.0022164, -0.0000820, -0.0015123, 0.0015754
5: 0.0150303, 0.0171875, 0.0150303, 0.0171875, -0.0015284, 0.0015923
6: 0.0035468, 0.0045961, 0.0035468, 0.0045961, -0.0007745, 0.0007434
7: -0.0129691, -0.0056961, -0.0129691, -0.0056961, -0.0051530, 0.0053683
8: 0.0064400, 0.0122101, 0.0064400, 0.0122101, -0.0040882, 0.0042590
9: 0.0093077, 0.0196857, 0.0093077, 0.0196857, -0.0073529, 0.0076601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016302, upper bound: 0.0015470
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016302, upper bound: 0.0015460
time: 1.18 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.60
Output dim: 2, lower bound: -0.0014245, upper bound: 0.0013730
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.60
Output dim: 2, lower bound: -0.0014245, upper bound: 0.0013730
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.60
Output dim: 2, lower bound: -0.0016302, upper bound: 0.0015470
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.60
Output dim: 2, lower bound: -0.0016302, upper bound: 0.0015460

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.77 + 183.13 = 186.90 seconds
