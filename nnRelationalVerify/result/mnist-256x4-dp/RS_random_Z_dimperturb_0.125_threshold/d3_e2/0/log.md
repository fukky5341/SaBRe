## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00026408


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0007074, 0.0007074)
1: (-0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002362, 0.0002362)
2: (0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0009069, 0.0009069)
3: (1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460)
4: (-0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001419, 0.0001419)
5: (0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005416, 0.0005416)
6: (-0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707)
7: (-0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011947, 0.0011947)
8: (-0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0015003, 0.0015003)
9: (-0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0007232, 0.0007232)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 1.50 = 2.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0004044, upper bound: 0.0004044

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004038, upper bound: 0.0004030
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004030, upper bound: 0.0004038
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 3, lower bound: -0.0004038, upper bound: 0.0004030
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 3, lower bound: -0.0004030, upper bound: 0.0004038

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0007028, 0.0007038
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002357, 0.0002361
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0009012, 0.0009026
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001412, 0.0001409
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005380, 0.0005387
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011943, 0.0011942
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014925, 0.0014895
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0007205, 0.0007222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0004030
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0004038, upper bound: 0.0003922
time: 0.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0007038, 0.0007028
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002361, 0.0002357
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0009026, 0.0009012
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001409, 0.0001412
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005387, 0.0005380
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011942, 0.0011943
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014895, 0.0014925
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0007222, 0.0007205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003840, upper bound: 0.0003841
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003840, upper bound: 0.0003841
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 3, lower bound: -0.0003930, upper bound: 0.0004030
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 3, lower bound: -0.0004038, upper bound: 0.0003922
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 3, lower bound: -0.0003840, upper bound: 0.0003841
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 3, lower bound: -0.0003840, upper bound: 0.0003841

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006826, 0.0006730
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002317, 0.0002270
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008703, 0.0008556
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001326, 0.0001354
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005220, 0.0005145
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011906, 0.0011917
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013926, 0.0014244
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006876, 0.0006706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003889, upper bound: 0.0004022
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003923, upper bound: 0.0004005
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006721, 0.0006827
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002266, 0.0002318
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008542, 0.0008705
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001354, 0.0001324
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005138, 0.0005221
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011917, 0.0011905
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014247, 0.0013896
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006690, 0.0006878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003841, upper bound: 0.0003746
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003841, upper bound: 0.0003746
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006968, 0.0006971
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002326, 0.0002328
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008919, 0.0008925
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001393, 0.0001392
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005332, 0.0005335
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011935, 0.0011935
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014705, 0.0014693
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0007097, 0.0007104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003746, upper bound: 0.0003841
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003840, upper bound: 0.0003752
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006981, 0.0007028
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002332, 0.0002357
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008939, 0.0009012
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001409, 0.0001396
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005342, 0.0005380
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011942, 0.0011936
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014895, 0.0014735
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0007120, 0.0007205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003813, upper bound: 0.0003834
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003833, upper bound: 0.0003811
time: 0.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -0.0003889, upper bound: 0.0004022
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -0.0003923, upper bound: 0.0004005
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -0.0003841, upper bound: 0.0003746
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -0.0003841, upper bound: 0.0003746
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -0.0003746, upper bound: 0.0003841
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -0.0003840, upper bound: 0.0003752
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -0.0003813, upper bound: 0.0003834
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -0.0003833, upper bound: 0.0003811

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006590, 0.0006484
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002257, 0.0002205
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008328, 0.0008164
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007426, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001254, 0.0001285
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005034, 0.0004950
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011878, 0.0011891
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013104, 0.0013460
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006464, 0.0006273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002503, upper bound: 0.0002510
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002503, upper bound: 0.0002510
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006597, 0.0006495
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002261, 0.0002211
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008339, 0.0008181
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007440, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001257, 0.0001287
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005040, 0.0004959
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011880, 0.0011892
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013143, 0.0013483
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006476, 0.0006294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003746, upper bound: 0.0003813
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003746, upper bound: 0.0003813
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006670, 0.0006775
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002243, 0.0002295
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008463, 0.0008625
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001339, 0.0001309
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005097, 0.0005180
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011911, 0.0011899
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014075, 0.0013725
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006599, 0.0006786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003794, upper bound: 0.0003688
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003794, upper bound: 0.0003687
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006670, 0.0006827
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002243, 0.0002318
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008463, 0.0008705
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001354, 0.0001309
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005097, 0.0005221
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011917, 0.0011899
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014247, 0.0013723
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006598, 0.0006878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003794, upper bound: 0.0003688
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003794, upper bound: 0.0003687
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006779, 0.0006670
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002297, 0.0002243
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008631, 0.0008463
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001309, 0.0001340
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005183, 0.0005097
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011899, 0.0011911
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013723, 0.0014088
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006793, 0.0006598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003700, upper bound: 0.0003834
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003739, upper bound: 0.0003811
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006666, 0.0006774
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002241, 0.0002294
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008457, 0.0008623
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001339, 0.0001308
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005094, 0.0005179
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011911, 0.0011898
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014071, 0.0013711
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006591, 0.0006784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003793, upper bound: 0.0003696
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003793, upper bound: 0.0003696
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006757, 0.0006800
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002260, 0.0002278
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008594, 0.0008659
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001344, 0.0001332
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005167, 0.0005200
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011914, 0.0011909
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014130, 0.0013989
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006729, 0.0006804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003767, upper bound: 0.0003788
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003767, upper bound: 0.0003788
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006746, 0.0006809
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002254, 0.0002282
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008576, 0.0008673
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001346, 0.0001328
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005157, 0.0005207
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011915, 0.0011908
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014160, 0.0013950
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006708, 0.0006820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003765
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003765
time: 0.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0002503, upper bound: 0.0002510
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0002503, upper bound: 0.0002510
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003746, upper bound: 0.0003813
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003746, upper bound: 0.0003813
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003794, upper bound: 0.0003688
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003794, upper bound: 0.0003687
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003794, upper bound: 0.0003688
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003794, upper bound: 0.0003687
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003700, upper bound: 0.0003834
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003739, upper bound: 0.0003811
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003793, upper bound: 0.0003696
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003793, upper bound: 0.0003696
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003767, upper bound: 0.0003788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003767, upper bound: 0.0003788
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003765
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003765

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006562, 0.0006442
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002245, 0.0002186
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008285, 0.0008100
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007384, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001242, 0.0001277
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005012, 0.0004917
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011873, 0.0011887
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012965, 0.0013367
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006414, 0.0006199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003690, upper bound: 0.0003767
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003690, upper bound: 0.0003767
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006544, 0.0006495
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002236, 0.0002211
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008257, 0.0008181
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007440, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001257, 0.0001271
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004998, 0.0004959
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011880, 0.0011885
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013143, 0.0013306
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006381, 0.0006294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003690, upper bound: 0.0003767
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003690, upper bound: 0.0003767
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006634, 0.0006731
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002225, 0.0002272
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008413, 0.0008562
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007446
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001328, 0.0001300
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005069, 0.0005146
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011906, 0.0011895
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013942, 0.0013619
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006533, 0.0006706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003682
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003788, upper bound: 0.0003653
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006670, 0.0006739
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002243, 0.0002276
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008463, 0.0008575
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001330, 0.0001309
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005097, 0.0005152
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011907, 0.0011899
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013969, 0.0013725
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006599, 0.0006720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003681
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003788, upper bound: 0.0003653
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006634, 0.0006783
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002225, 0.0002295
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008413, 0.0008642
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007446
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001342, 0.0001300
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005069, 0.0005186
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011912, 0.0011894
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014115, 0.0013618
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006532, 0.0006798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003682
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003788, upper bound: 0.0003653
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006670, 0.0006791
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002243, 0.0002299
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008463, 0.0008654
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001345, 0.0001309
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005097, 0.0005193
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011913, 0.0011899
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014142, 0.0013723
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006598, 0.0006812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003681
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003788, upper bound: 0.0003653
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006542, 0.0006424
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002235, 0.0002178
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008254, 0.0008072
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007362, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001237, 0.0001271
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004996, 0.0004903
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011871, 0.0011885
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012905, 0.0013299
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006378, 0.0006167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003653, upper bound: 0.0003788
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003652, upper bound: 0.0003788
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006547, 0.0006433
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002238, 0.0002182
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008260, 0.0008085
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007373, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001239, 0.0001272
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004999, 0.0004910
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011872, 0.0011886
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012934, 0.0013314
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006385, 0.0006183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003681, upper bound: 0.0003765
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003682, upper bound: 0.0003765
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006630, 0.0006735
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002223, 0.0002274
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008407, 0.0008568
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007441
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001329, 0.0001299
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005066, 0.0005148
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011906, 0.0011894
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013954, 0.0013605
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006526, 0.0006712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003766, upper bound: 0.0003690
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003656
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006666, 0.0006738
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002241, 0.0002276
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008457, 0.0008573
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001330, 0.0001308
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005094, 0.0005151
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011907, 0.0011898
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013966, 0.0013711
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006591, 0.0006718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003766, upper bound: 0.0003690
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003656
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006722, 0.0006765
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002241, 0.0002260
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008543, 0.0008609
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007449
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001334, 0.0001322
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005139, 0.0005173
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011910, 0.0011905
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014016, 0.0013873
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006662, 0.0006738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003653, upper bound: 0.0003788
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003766, upper bound: 0.0003690
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006757, 0.0006764
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002260, 0.0002259
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008594, 0.0008608
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001334, 0.0001332
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005167, 0.0005172
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011910, 0.0011909
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014013, 0.0013989
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006729, 0.0006737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003652, upper bound: 0.0003788
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003766, upper bound: 0.0003690
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006710, 0.0006769
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002236, 0.0002261
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008525, 0.0008615
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007435
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001335, 0.0001319
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005130, 0.0005176
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011911, 0.0011904
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014028, 0.0013834
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006641, 0.0006745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003681, upper bound: 0.0003765
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003656
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006746, 0.0006773
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002254, 0.0002264
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008576, 0.0008622
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001337, 0.0001328
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005157, 0.0005179
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011911, 0.0011908
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0014043, 0.0013950
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006708, 0.0006753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003682, upper bound: 0.0003765
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003656
time: 0.55 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003690, upper bound: 0.0003767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003690, upper bound: 0.0003767
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003690, upper bound: 0.0003767
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003690, upper bound: 0.0003767
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003682
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003788, upper bound: 0.0003653
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003681
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003788, upper bound: 0.0003653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003682
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003788, upper bound: 0.0003653
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003681
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003788, upper bound: 0.0003653
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003653, upper bound: 0.0003788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003652, upper bound: 0.0003788
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003681, upper bound: 0.0003765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003682, upper bound: 0.0003765
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003766, upper bound: 0.0003690
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003656
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003766, upper bound: 0.0003690
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003656
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003653, upper bound: 0.0003788
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003766, upper bound: 0.0003690
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003652, upper bound: 0.0003788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003766, upper bound: 0.0003690
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003681, upper bound: 0.0003765
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003656
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003682, upper bound: 0.0003765
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 3, lower bound: -0.0003787, upper bound: 0.0003656

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006527, 0.0006402
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002226, 0.0002164
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008237, 0.0008044
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007330, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001232, 0.0001268
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004985, 0.0004886
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011869, 0.0011883
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012828, 0.0013247
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006342, 0.0006118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 1.81 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003674
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003614, upper bound: 0.0003765
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006562, 0.0006407
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002245, 0.0002167
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008285, 0.0008051
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007336, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001233, 0.0001277
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005012, 0.0004890
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011869, 0.0011887
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012845, 0.0013367
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006414, 0.0006127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 1.79 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003674
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003614, upper bound: 0.0003765
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006509, 0.0006455
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002217, 0.0002189
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008208, 0.0008125
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007385, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001247, 0.0001262
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004970, 0.0004928
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011875, 0.0011881
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013006, 0.0013186
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006309, 0.0006213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 1.95 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003679
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003593, upper bound: 0.0003679
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006544, 0.0006460
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002236, 0.0002191
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008257, 0.0008133
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007392, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001248, 0.0001271
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004998, 0.0004932
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011875, 0.0011885
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013023, 0.0013306
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006381, 0.0006222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 1.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003674
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003614, upper bound: 0.0003765
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006398, 0.0006490
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002163, 0.0002208
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008038, 0.0008180
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007438, 0.0007325
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001257, 0.0001230
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004883, 0.0004956
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011879, 0.0011868
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013124, 0.0012816
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006112, 0.0006276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 1.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003763, upper bound: 0.0003614
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003669, upper bound: 0.0003680
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006394, 0.0006495
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002161, 0.0002210
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008032, 0.0008187
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007444, 0.0007321
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001258, 0.0001229
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004880, 0.0004959
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011879, 0.0011868
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013139, 0.0012804
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006106, 0.0006285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 1.80 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003701, upper bound: 0.0003550
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003700, upper bound: 0.0003562
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006433, 0.0006495
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002182, 0.0002210
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008086, 0.0008187
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007444, 0.0007374
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001258, 0.0001240
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004910, 0.0004959
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011879, 0.0011872
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013139, 0.0012936
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006184, 0.0006285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 1.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002995, upper bound: 0.0002928
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002989, upper bound: 0.0002946
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006430, 0.0006503
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002180, 0.0002214
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008081, 0.0008199
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007453, 0.0007369
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001261, 0.0001238
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004907, 0.0004965
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011880, 0.0011872
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013166, 0.0012924
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006177, 0.0006299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 1.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003389, upper bound: 0.0003566
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003288
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006398, 0.0006544
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002162, 0.0002232
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008037, 0.0008262
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007325
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001272, 0.0001230
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004882, 0.0004997
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011885, 0.0011868
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013302, 0.0012815
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006111, 0.0006371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 1.79 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003068, upper bound: 0.0003071
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003068, upper bound: 0.0003071
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006389, 0.0006548
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002158, 0.0002234
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008023, 0.0008269
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007314
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001274, 0.0001228
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004875, 0.0005001
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011886, 0.0011867
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013317, 0.0012785
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006095, 0.0006379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 1.83 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003234
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003248, upper bound: 0.0003250
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006433, 0.0006548
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002182, 0.0002234
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008085, 0.0008269
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007373
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001274, 0.0001239
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004910, 0.0005001
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011886, 0.0011872
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013317, 0.0012934
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006183, 0.0006380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 1.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003068, upper bound: 0.0003071
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003068, upper bound: 0.0003071
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006424, 0.0006556
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002178, 0.0002238
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008072, 0.0008281
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007362
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001276, 0.0001237
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004903, 0.0005007
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011887, 0.0011871
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013344, 0.0012905
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006167, 0.0006394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 1.80 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002994, upper bound: 0.0002940
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002986, upper bound: 0.0002948
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006507, 0.0006389
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002216, 0.0002158
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008206, 0.0008025
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007315, 0.0007458
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001228, 0.0001262
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004969, 0.0004876
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011867, 0.0011881
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012787, 0.0013180
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006306, 0.0006096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 1.79 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003250, upper bound: 0.0003248
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003235, upper bound: 0.0003274
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006542, 0.0006389
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002235, 0.0002158
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008254, 0.0008023
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007314, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001228, 0.0001271
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004996, 0.0004875
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011867, 0.0011885
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012785, 0.0013299
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006378, 0.0006095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 1.80 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003250, upper bound: 0.0003248
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003234, upper bound: 0.0003274
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006511, 0.0006393
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002218, 0.0002160
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008212, 0.0008030
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007319, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001229, 0.0001263
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004972, 0.0004879
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011868, 0.0011881
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012800, 0.0013194
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006314, 0.0006103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 1.79 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003267, upper bound: 0.0003245
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003241, upper bound: 0.0003266
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006547, 0.0006398
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002238, 0.0002162
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008260, 0.0008037
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007325, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001230, 0.0001272
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004999, 0.0004882
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011868, 0.0011886
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012815, 0.0013314
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006385, 0.0006111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 1.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003593, upper bound: 0.0003679
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003576, upper bound: 0.0003679
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006394, 0.0006505
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002161, 0.0002215
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008032, 0.0008203
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007456, 0.0007320
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001261, 0.0001229
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004880, 0.0004967
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011881, 0.0011868
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013173, 0.0012803
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006105, 0.0006303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003679, upper bound: 0.0003593
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003679, upper bound: 0.0003602
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006382, 0.0006499
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002155, 0.0002212
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008013, 0.0008193
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007448, 0.0007306
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001259, 0.0001226
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004870, 0.0004962
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011880, 0.0011866
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013151, 0.0012763
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006083, 0.0006291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 1.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002982, upper bound: 0.0002961
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002970, upper bound: 0.0002969
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006429, 0.0006509
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002180, 0.0002217
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008080, 0.0008208
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007369
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001262, 0.0001238
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004907, 0.0004970
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011881, 0.0011872
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013186, 0.0012922
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006176, 0.0006309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 1.92 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003614
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003674, upper bound: 0.0003688
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006417, 0.0006502
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002174, 0.0002214
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008061, 0.0008198
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007452, 0.0007354
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001260, 0.0001235
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004898, 0.0004965
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011880, 0.0011870
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013163, 0.0012882
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006155, 0.0006297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002982, upper bound: 0.0002961
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002970, upper bound: 0.0002969
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006503, 0.0006443
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002214, 0.0002183
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008199, 0.0008107
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007370, 0.0007453
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001243, 0.0001261
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004965, 0.0004918
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011873, 0.0011880
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012965, 0.0013166
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006299, 0.0006191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 1.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003288, upper bound: 0.0003687
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003566, upper bound: 0.0003389
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006407, 0.0006558
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002167, 0.0002239
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008051, 0.0008285
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007336
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001276, 0.0001233
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004890, 0.0005009
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011887, 0.0011869
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013351, 0.0012845
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006127, 0.0006398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 1.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003614
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003674, upper bound: 0.0003688
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006538, 0.0006442
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002233, 0.0002182
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008248, 0.0008105
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007370, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001243, 0.0001270
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004993, 0.0004917
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011873, 0.0011885
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012962, 0.0013286
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006371, 0.0006190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 1.84 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002948, upper bound: 0.0002986
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002940, upper bound: 0.0002994
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006442, 0.0006562
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002186, 0.0002241
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008100, 0.0008290
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007384
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001278, 0.0001242
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004917, 0.0005012
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011887, 0.0011873
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013363, 0.0012965
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006199, 0.0006404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 1.83 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003299, upper bound: 0.0003599
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003675, upper bound: 0.0003381
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006495, 0.0006446
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002210, 0.0002185
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008187, 0.0008112
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007375, 0.0007444
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001244, 0.0001258
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004959, 0.0004921
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011874, 0.0011879
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012977, 0.0013139
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006285, 0.0006198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 1.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002946, upper bound: 0.0002989
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002928, upper bound: 0.0002995
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006395, 0.0006552
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002161, 0.0002236
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008033, 0.0008275
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007322
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001275, 0.0001230
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004881, 0.0005004
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011886, 0.0011868
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013329, 0.0012807
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006107, 0.0006386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 1.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003699, upper bound: 0.0003560
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003698, upper bound: 0.0003565
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006530, 0.0006451
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002230, 0.0002187
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008235, 0.0008119
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007381, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001246, 0.0001267
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004987, 0.0004924
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011874, 0.0011884
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012992, 0.0013259
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006356, 0.0006206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 1.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003267, upper bound: 0.0003245
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003240, upper bound: 0.0003266
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006430, 0.0006555
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002181, 0.0002238
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008082, 0.0008280
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007370
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001276, 0.0001239
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004908, 0.0005007
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011887, 0.0011872
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013340, 0.0012926
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006178, 0.0006392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 1.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003786, upper bound: 0.0003603
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003674, upper bound: 0.0003655
time: 0.59 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.54 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003674
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003614, upper bound: 0.0003765
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003674
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003614, upper bound: 0.0003765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003602, upper bound: 0.0003679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003593, upper bound: 0.0003679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003688, upper bound: 0.0003674
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003614, upper bound: 0.0003765
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003763, upper bound: 0.0003614
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003669, upper bound: 0.0003680
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003701, upper bound: 0.0003550
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003700, upper bound: 0.0003562
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002995, upper bound: 0.0002928
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002989, upper bound: 0.0002946
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003389, upper bound: 0.0003566
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003687, upper bound: 0.0003288
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003068, upper bound: 0.0003071
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003068, upper bound: 0.0003071
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003274, upper bound: 0.0003234
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003248, upper bound: 0.0003250
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003068, upper bound: 0.0003071
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003068, upper bound: 0.0003071
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002994, upper bound: 0.0002940
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002986, upper bound: 0.0002948
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003250, upper bound: 0.0003248
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003235, upper bound: 0.0003274
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003250, upper bound: 0.0003248
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003234, upper bound: 0.0003274
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003267, upper bound: 0.0003245
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003241, upper bound: 0.0003266
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003593, upper bound: 0.0003679
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003576, upper bound: 0.0003679
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003679, upper bound: 0.0003593
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003679, upper bound: 0.0003602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002982, upper bound: 0.0002961
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002970, upper bound: 0.0002969
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003614
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003674, upper bound: 0.0003688
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002982, upper bound: 0.0002961
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002970, upper bound: 0.0002969
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003288, upper bound: 0.0003687
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003566, upper bound: 0.0003389
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003765, upper bound: 0.0003614
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003674, upper bound: 0.0003688
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002948, upper bound: 0.0002986
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002940, upper bound: 0.0002994
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003299, upper bound: 0.0003599
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003675, upper bound: 0.0003381
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002946, upper bound: 0.0002989
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0002928, upper bound: 0.0002995
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003699, upper bound: 0.0003560
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003698, upper bound: 0.0003565
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003267, upper bound: 0.0003245
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003240, upper bound: 0.0003266
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003786, upper bound: 0.0003603
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.54
Output dim: 3, lower bound: -0.0003674, upper bound: 0.0003655

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006371, 0.0006309
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002112, 0.0002081
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007978, 0.0007882
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007165, 0.0007241
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001193, 0.0001211
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004862, 0.0004813
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011860, 0.0011867
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012326, 0.0012534
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005889, 0.0005778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003601, upper bound: 0.0003589
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003592, upper bound: 0.0003591
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006435, 0.0006255
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002143, 0.0002055
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008075, 0.0007798
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007099, 0.0007318
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001177, 0.0001229
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004912, 0.0004770
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011854, 0.0011875
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012145, 0.0012745
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006001, 0.0005681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003334, upper bound: 0.0003673
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003518, upper bound: 0.0003297
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006408, 0.0006314
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002133, 0.0002084
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008034, 0.0007890
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007171, 0.0007294
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001194, 0.0001220
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004891, 0.0004817
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011860, 0.0011872
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012343, 0.0012660
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005964, 0.0005787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003064, upper bound: 0.0003070
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003064, upper bound: 0.0003070
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006469, 0.0006260
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002163, 0.0002057
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008128, 0.0007806
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007105, 0.0007368
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001179, 0.0001238
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004939, 0.0004774
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011854, 0.0011879
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012163, 0.0012865
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006073, 0.0005690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003531, upper bound: 0.0003678
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003528, upper bound: 0.0003678
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006317, 0.0006288
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002117, 0.0002103
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007913, 0.0007868
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007190, 0.0007231
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001199, 0.0001207
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004819, 0.0004796
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011855, 0.0011859
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012448, 0.0012545
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005967, 0.0005915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003294, upper bound: 0.0003591
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003513, upper bound: 0.0003222
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006345, 0.0006242
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002131, 0.0002080
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007957, 0.0007798
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007135, 0.0007266
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001186, 0.0001215
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004841, 0.0004760
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011850, 0.0011862
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012297, 0.0012640
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006018, 0.0005835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003592, upper bound: 0.0003591
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003528, upper bound: 0.0003678
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006390, 0.0006378
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002124, 0.0002107
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008006, 0.0007988
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007224, 0.0007272
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001213, 0.0001215
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004876, 0.0004867
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011868, 0.0011869
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012556, 0.0012600
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005932, 0.0005900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003064, upper bound: 0.0003070
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003064, upper bound: 0.0003070
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006451, 0.0006321
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002154, 0.0002080
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008100, 0.0007900
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007155, 0.0007346
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001196, 0.0001233
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004924, 0.0004822
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011861, 0.0011877
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012365, 0.0012804
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006040, 0.0005798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002933, upper bound: 0.0002978
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002923, upper bound: 0.0002986
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006250, 0.0006398
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002052, 0.0002125
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007791, 0.0008018
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007273, 0.0007093
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001218, 0.0001176
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004766, 0.0004883
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011870, 0.0011853
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012622, 0.0012128
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005672, 0.0005936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003055, upper bound: 0.0003070
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003055, upper bound: 0.0003070
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006305, 0.0006338
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002080, 0.0002096
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007876, 0.0007927
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007201, 0.0007160
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001201, 0.0001192
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004810, 0.0004836
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011863, 0.0011859
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012424, 0.0012314
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005771, 0.0005830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002957, upper bound: 0.0002927
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002956, upper bound: 0.0002944
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006183, 0.0006331
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002051, 0.0002124
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007707, 0.0007935
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007249, 0.0007068
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001211, 0.0001169
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004713, 0.0004830
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011860, 0.0011843
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012593, 0.0012098
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005728, 0.0005993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003006, upper bound: 0.0002992
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003006, upper bound: 0.0002992
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006231, 0.0006300
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002075, 0.0002109
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007781, 0.0007887
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007211, 0.0007127
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001202, 0.0001183
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004751, 0.0004806
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011857, 0.0011848
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012489, 0.0012258
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005814, 0.0005937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003203, upper bound: 0.0003156
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003172, upper bound: 0.0003176
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006061, 0.0006345
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002004, 0.0002138
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007512, 0.0007957
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007267, 0.0006932
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001215, 0.0001133
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004618, 0.0004841
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011862, 0.0011829
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012641, 0.0011697
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005528, 0.0006018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002355, upper bound: 0.0002343
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002355, upper bound: 0.0002343
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006284, 0.0006495
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002110, 0.0002210
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007856, 0.0008187
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007444, 0.0007198
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001258, 0.0001197
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004792, 0.0004959
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011879, 0.0011855
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013139, 0.0012437
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005917, 0.0006285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002714, upper bound: 0.0002666
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002714, upper bound: 0.0002666
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006195, 0.0006010
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002139, 0.0002045
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007690, 0.0007408
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007072, 0.0007305
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001109, 0.0001161
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004721, 0.0004576
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011826, 0.0011847
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011449, 0.0012092
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005872, 0.0005529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003304, upper bound: 0.0003471
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003307, upper bound: 0.0003476
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0005938, 0.0006263
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002012, 0.0002169
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007296, 0.0007797
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007380, 0.0006988
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001181, 0.0001088
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004519, 0.0004775
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011856, 0.0011817
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012291, 0.0011221
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005404, 0.0005980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002798, upper bound: 0.0002795
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002798, upper bound: 0.0002795
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006387, 0.0006546
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002158, 0.0002234
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008022, 0.0008268
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007314
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001273, 0.0001228
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004874, 0.0005000
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011886, 0.0011867
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013318, 0.0012785
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006096, 0.0006380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002798, upper bound: 0.0002796
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002798, upper bound: 0.0002796
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006398, 0.0006533
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002162, 0.0002228
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008037, 0.0008247
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007325
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001269, 0.0001230
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004882, 0.0004989
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011884, 0.0011868
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013272, 0.0012815
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006111, 0.0006356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002999, upper bound: 0.0003002
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002997, upper bound: 0.0003003
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006014, 0.0006341
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001963, 0.0002118
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007448, 0.0007950
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007214, 0.0006835
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001214, 0.0001121
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004581, 0.0004838
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011861, 0.0011823
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012626, 0.0011537
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005429, 0.0006010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002854, upper bound: 0.0002827
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002854, upper bound: 0.0002827
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006179, 0.0006548
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002044, 0.0002234
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007701, 0.0008269
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007035
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001274, 0.0001168
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004710, 0.0005001
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011886, 0.0011842
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013317, 0.0012085
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005721, 0.0006379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003247, upper bound: 0.0003229
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003229, upper bound: 0.0003249
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006422, 0.0006551
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002178, 0.0002237
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008070, 0.0008275
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007363
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001275, 0.0001237
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004902, 0.0005004
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011886, 0.0011871
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013334, 0.0012905
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006167, 0.0006389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002846, upper bound: 0.0002833
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002834, upper bound: 0.0002847
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006433, 0.0006538
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002182, 0.0002230
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008085, 0.0008254
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007373
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001271, 0.0001239
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004910, 0.0004993
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011884, 0.0011872
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013288, 0.0012934
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006183, 0.0006364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002846, upper bound: 0.0002833
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002834, upper bound: 0.0002847
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006074, 0.0006408
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002010, 0.0002166
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007531, 0.0008053
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007329, 0.0006947
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001233, 0.0001136
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004628, 0.0004890
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011869, 0.0011830
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012848, 0.0011739
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005550, 0.0006129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002761, upper bound: 0.0002726
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002761, upper bound: 0.0002726
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006274, 0.0006556
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002106, 0.0002238
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007842, 0.0008281
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007186
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001276, 0.0001194
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004785, 0.0005007
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011887, 0.0011854
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013344, 0.0012406
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005900, 0.0006394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002716, upper bound: 0.0002666
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002716, upper bound: 0.0002666
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006085, 0.0006179
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001998, 0.0002044
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007556, 0.0007702
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007036, 0.0006921
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001168, 0.0001141
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004636, 0.0004711
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011842, 0.0011831
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012088, 0.0011773
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005554, 0.0005723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003176, upper bound: 0.0003172
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003177, upper bound: 0.0003171
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006297, 0.0006389
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002102, 0.0002158
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007883, 0.0008025
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007315, 0.0007180
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001228, 0.0001202
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004803, 0.0004876
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011867, 0.0011856
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012787, 0.0012480
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005932, 0.0006096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002727, upper bound: 0.0002761
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002726, upper bound: 0.0002761
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006137, 0.0006179
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002028, 0.0002044
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007628, 0.0007701
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007035, 0.0006995
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001168, 0.0001154
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004677, 0.0004710
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011842, 0.0011837
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012085, 0.0011949
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005663, 0.0005721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002839, upper bound: 0.0002842
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002839, upper bound: 0.0002842
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006332, 0.0006389
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002121, 0.0002158
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007931, 0.0008023
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007314, 0.0007228
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001228, 0.0001211
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004831, 0.0004875
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011867, 0.0011860
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012785, 0.0012600
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006004, 0.0006095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002999, upper bound: 0.0003066
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003021, upper bound: 0.0003057
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006091, 0.0006183
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002001, 0.0002046
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007566, 0.0007708
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007041, 0.0006929
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001169, 0.0001143
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004641, 0.0004714
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011843, 0.0011832
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012100, 0.0011794
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005566, 0.0005729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003266, upper bound: 0.0003223
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003242, upper bound: 0.0003244
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006301, 0.0006393
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002104, 0.0002160
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007889, 0.0008030
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007319, 0.0007185
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001229, 0.0001203
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004807, 0.0004879
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011868, 0.0011857
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012800, 0.0012495
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005940, 0.0006103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003240, upper bound: 0.0003242
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003220, upper bound: 0.0003265
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006365, 0.0006234
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002144, 0.0002076
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007979, 0.0007785
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007130, 0.0007299
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001183, 0.0001220
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004856, 0.0004754
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011849, 0.0011864
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012269, 0.0012708
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006065, 0.0005819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003192, upper bound: 0.0003168
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003162, upper bound: 0.0003196
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006383, 0.0006194
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002152, 0.0002057
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008009, 0.0007724
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007082, 0.0007318
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001172, 0.0001225
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004871, 0.0004722
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011844, 0.0011866
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012136, 0.0012768
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006094, 0.0005748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003243, upper bound: 0.0003593
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003486, upper bound: 0.0003265
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006197, 0.0006341
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002059, 0.0002129
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007729, 0.0007951
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007262, 0.0007086
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001214, 0.0001173
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004725, 0.0004838
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011861, 0.0011845
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012627, 0.0012147
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005754, 0.0006011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003193, upper bound: 0.0003171
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003192
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006230, 0.0006310
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002075, 0.0002114
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007780, 0.0007903
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007223, 0.0007126
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001205, 0.0001182
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004751, 0.0004814
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011858, 0.0011848
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012523, 0.0012257
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005813, 0.0005955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003221, upper bound: 0.0003513
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003591, upper bound: 0.0003294
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006021, 0.0006349
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001980, 0.0002140
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007459, 0.0007963
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007272, 0.0006872
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001216, 0.0001123
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004586, 0.0004844
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011862, 0.0011824
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012653, 0.0011561
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005441, 0.0006025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002981, upper bound: 0.0002940
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002944, upper bound: 0.0002960
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006232, 0.0006499
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002083, 0.0002212
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007783, 0.0008193
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007448, 0.0007129
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001259, 0.0001183
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004752, 0.0004962
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011880, 0.0011849
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013151, 0.0012264
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005817, 0.0006291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002688, upper bound: 0.0002696
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002688, upper bound: 0.0002696
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006282, 0.0006416
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002072, 0.0002134
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007840, 0.0008047
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007296, 0.0007140
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001224, 0.0001184
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004792, 0.0004897
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011872, 0.0011857
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012684, 0.0012242
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005740, 0.0005969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003054, upper bound: 0.0003069
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003054, upper bound: 0.0003069
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006336, 0.0006353
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002098, 0.0002103
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007923, 0.0007950
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007219, 0.0007205
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001205, 0.0001200
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004834, 0.0004848
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011865, 0.0011863
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012474, 0.0012420
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005835, 0.0005856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003591, upper bound: 0.0003592
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003589, upper bound: 0.0003601
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006073, 0.0006352
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002009, 0.0002142
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007529, 0.0007968
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007276, 0.0006945
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001217, 0.0001136
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004626, 0.0004847
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011863, 0.0011830
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012664, 0.0011735
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005548, 0.0006031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002688, upper bound: 0.0002696
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002688, upper bound: 0.0002696
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006267, 0.0006502
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002102, 0.0002214
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007831, 0.0008198
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007452, 0.0007178
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001260, 0.0001192
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004780, 0.0004965
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011880, 0.0011853
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013163, 0.0012384
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005889, 0.0006297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002350, upper bound: 0.0002349
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002350, upper bound: 0.0002349
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006263, 0.0005946
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002169, 0.0002012
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007797, 0.0007309
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0006982, 0.0007380
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001091, 0.0001181
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004775, 0.0004526
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011818, 0.0011856
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011235, 0.0012291
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005980, 0.0005415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003210, upper bound: 0.0003606
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003211, upper bound: 0.0003606
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006010, 0.0006196
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002045, 0.0002134
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007408, 0.0007693
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007284, 0.0007072
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001162, 0.0001109
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004576, 0.0004722
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011848, 0.0011826
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012067, 0.0011449
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005529, 0.0005860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003476, upper bound: 0.0003307
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003471, upper bound: 0.0003304
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006260, 0.0006476
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002057, 0.0002156
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007806, 0.0008139
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007344, 0.0007105
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001241, 0.0001179
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004774, 0.0004945
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011880, 0.0011854
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012884, 0.0012163
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005690, 0.0006076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003264, upper bound: 0.0003227
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003238, upper bound: 0.0003243
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006314, 0.0006411
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002084, 0.0002124
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007890, 0.0008039
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007266, 0.0007171
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001222, 0.0001194
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004817, 0.0004893
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011872, 0.0011860
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012667, 0.0012343
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005787, 0.0005960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002955, upper bound: 0.0002957
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002954, upper bound: 0.0002966
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006160, 0.0006293
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002052, 0.0002110
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007664, 0.0007877
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007189, 0.0007052
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001200, 0.0001161
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004695, 0.0004800
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011856, 0.0011840
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012467, 0.0012027
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005704, 0.0005925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002727, upper bound: 0.0002761
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002727, upper bound: 0.0002761
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006389, 0.0006442
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002162, 0.0002182
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008018, 0.0008105
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007370, 0.0007326
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001243, 0.0001227
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004875, 0.0004917
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011873, 0.0011867
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012962, 0.0012788
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006104, 0.0006190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002343, upper bound: 0.0002356
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002343, upper bound: 0.0002356
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006188, 0.0006066
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002136, 0.0002071
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007680, 0.0007493
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007127, 0.0007297
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001125, 0.0001159
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004716, 0.0004620
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011832, 0.0011847
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011634, 0.0012070
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005860, 0.0005628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002795, upper bound: 0.0002798
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002795, upper bound: 0.0002798
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0005950, 0.0006334
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002018, 0.0002201
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007315, 0.0007905
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007452, 0.0007003
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001202, 0.0001091
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004529, 0.0004830
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011864, 0.0011819
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012526, 0.0011262
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005426, 0.0006105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003592, upper bound: 0.0003299
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003591, upper bound: 0.0003294
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006124, 0.0006298
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002030, 0.0002112
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007617, 0.0007884
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007195, 0.0006998
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001202, 0.0001152
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004667, 0.0004804
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011856, 0.0011836
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012482, 0.0011903
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005624, 0.0005933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002844, upper bound: 0.0002894
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002831, upper bound: 0.0002895
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006345, 0.0006446
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002138, 0.0002185
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007957, 0.0008112
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007375, 0.0007267
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001244, 0.0001215
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004841, 0.0004921
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011874, 0.0011862
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012977, 0.0012641
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006018, 0.0006198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002343, upper bound: 0.0002355
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002343, upper bound: 0.0002355
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006196, 0.0006385
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002058, 0.0002150
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007728, 0.0008017
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007308, 0.0007085
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001227, 0.0001173
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004724, 0.0004872
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011867, 0.0011844
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012771, 0.0012144
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005753, 0.0006088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003007, upper bound: 0.0002989
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003007, upper bound: 0.0002989
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006231, 0.0006358
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002075, 0.0002137
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007782, 0.0007976
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007275, 0.0007127
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001219, 0.0001183
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004752, 0.0004851
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011863, 0.0011849
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012682, 0.0012261
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005815, 0.0006040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003006, upper bound: 0.0002991
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003006, upper bound: 0.0002991
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006129, 0.0006243
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002024, 0.0002071
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007615, 0.0007800
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007095, 0.0006985
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001186, 0.0001152
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004671, 0.0004761
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011850, 0.0011837
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012301, 0.0011922
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005648, 0.0005837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003002, upper bound: 0.0003056
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003025, upper bound: 0.0003042
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006320, 0.0006451
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002115, 0.0002187
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007913, 0.0008119
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007381, 0.0007213
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001246, 0.0001207
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004821, 0.0004924
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011874, 0.0011859
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012992, 0.0012560
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005983, 0.0006206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002833, upper bound: 0.0002846
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002833, upper bound: 0.0002846
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006279, 0.0006473
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002070, 0.0002154
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007837, 0.0008134
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007340, 0.0007137
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001240, 0.0001184
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004790, 0.0004942
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011879, 0.0011857
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012873, 0.0012233
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005736, 0.0006070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003272, upper bound: 0.0003223
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003232
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006337, 0.0006404
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002098, 0.0002121
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007925, 0.0008028
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007257, 0.0007207
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001220, 0.0001200
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004835, 0.0004888
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011871, 0.0011863
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012643, 0.0012424
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005837, 0.0005947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003075, upper bound: 0.0003048
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003075, upper bound: 0.0003048
time: 0.57 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.85 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003601, upper bound: 0.0003589
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003592, upper bound: 0.0003591
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003334, upper bound: 0.0003673
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003518, upper bound: 0.0003297
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003064, upper bound: 0.0003070
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003064, upper bound: 0.0003070
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003531, upper bound: 0.0003678
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003528, upper bound: 0.0003678
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003294, upper bound: 0.0003591
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003513, upper bound: 0.0003222
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003592, upper bound: 0.0003591
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003528, upper bound: 0.0003678
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003064, upper bound: 0.0003070
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003064, upper bound: 0.0003070
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002933, upper bound: 0.0002978
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002923, upper bound: 0.0002986
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003055, upper bound: 0.0003070
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003055, upper bound: 0.0003070
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002957, upper bound: 0.0002927
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002956, upper bound: 0.0002944
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003006, upper bound: 0.0002992
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003006, upper bound: 0.0002992
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003203, upper bound: 0.0003156
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003172, upper bound: 0.0003176
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002355, upper bound: 0.0002343
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002355, upper bound: 0.0002343
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002714, upper bound: 0.0002666
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002714, upper bound: 0.0002666
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003304, upper bound: 0.0003471
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003307, upper bound: 0.0003476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002798, upper bound: 0.0002795
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002798, upper bound: 0.0002795
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002798, upper bound: 0.0002796
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002798, upper bound: 0.0002796
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002999, upper bound: 0.0003002
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002997, upper bound: 0.0003003
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002854, upper bound: 0.0002827
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002854, upper bound: 0.0002827
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003247, upper bound: 0.0003229
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003229, upper bound: 0.0003249
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002846, upper bound: 0.0002833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002834, upper bound: 0.0002847
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002846, upper bound: 0.0002833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002834, upper bound: 0.0002847
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002761, upper bound: 0.0002726
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002761, upper bound: 0.0002726
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002716, upper bound: 0.0002666
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002716, upper bound: 0.0002666
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003176, upper bound: 0.0003172
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003177, upper bound: 0.0003171
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002727, upper bound: 0.0002761
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002726, upper bound: 0.0002761
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002839, upper bound: 0.0002842
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002839, upper bound: 0.0002842
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002999, upper bound: 0.0003066
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003021, upper bound: 0.0003057
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003266, upper bound: 0.0003223
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003242, upper bound: 0.0003244
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003240, upper bound: 0.0003242
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003220, upper bound: 0.0003265
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003192, upper bound: 0.0003168
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003162, upper bound: 0.0003196
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003243, upper bound: 0.0003593
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003486, upper bound: 0.0003265
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003193, upper bound: 0.0003171
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003161, upper bound: 0.0003192
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003221, upper bound: 0.0003513
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003591, upper bound: 0.0003294
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002981, upper bound: 0.0002940
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002944, upper bound: 0.0002960
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002688, upper bound: 0.0002696
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002688, upper bound: 0.0002696
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003054, upper bound: 0.0003069
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003054, upper bound: 0.0003069
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003591, upper bound: 0.0003592
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003589, upper bound: 0.0003601
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002688, upper bound: 0.0002696
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002688, upper bound: 0.0002696
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002350, upper bound: 0.0002349
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002350, upper bound: 0.0002349
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003210, upper bound: 0.0003606
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003211, upper bound: 0.0003606
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003476, upper bound: 0.0003307
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003471, upper bound: 0.0003304
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003264, upper bound: 0.0003227
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003238, upper bound: 0.0003243
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002955, upper bound: 0.0002957
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002954, upper bound: 0.0002966
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002727, upper bound: 0.0002761
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002727, upper bound: 0.0002761
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002343, upper bound: 0.0002356
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002343, upper bound: 0.0002356
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002795, upper bound: 0.0002798
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002795, upper bound: 0.0002798
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003592, upper bound: 0.0003299
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003591, upper bound: 0.0003294
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002844, upper bound: 0.0002894
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002831, upper bound: 0.0002895
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002343, upper bound: 0.0002355
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002343, upper bound: 0.0002355
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003007, upper bound: 0.0002989
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003007, upper bound: 0.0002989
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003006, upper bound: 0.0002991
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003006, upper bound: 0.0002991
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003002, upper bound: 0.0003056
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003025, upper bound: 0.0003042
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002833, upper bound: 0.0002846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0002833, upper bound: 0.0002846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003272, upper bound: 0.0003223
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003239, upper bound: 0.0003232
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003075, upper bound: 0.0003048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 3, lower bound: -0.0003075, upper bound: 0.0003048

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006323, 0.0006238
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002120, 0.0002078
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007923, 0.0007792
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007135, 0.0007239
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001185, 0.0001209
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004824, 0.0004757
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011849, 0.0011859
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012283, 0.0012567
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005979, 0.0005827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002995, upper bound: 0.0002999
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002995, upper bound: 0.0002999
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006363, 0.0006191
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002140, 0.0002056
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007985, 0.0007719
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007078, 0.0007288
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001171, 0.0001221
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004856, 0.0004720
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011844, 0.0011864
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012126, 0.0012701
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006050, 0.0005743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002857, upper bound: 0.0002862
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002843, upper bound: 0.0002864
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006299, 0.0005909
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002187, 0.0001996
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007852, 0.0007252
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0006948, 0.0007423
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001080, 0.0001192
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004803, 0.0004496
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011814, 0.0011860
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011111, 0.0012410
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006043, 0.0005349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003013, upper bound: 0.0003026
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003013, upper bound: 0.0003026
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006035, 0.0006133
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002057, 0.0002105
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007445, 0.0007596
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007220, 0.0007101
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001144, 0.0001116
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004595, 0.0004672
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011840, 0.0011829
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011856, 0.0011530
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005573, 0.0005747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002792, upper bound: 0.0002794
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002792, upper bound: 0.0002794
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006552, 0.0006410
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002241, 0.0002170
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008270, 0.0008059
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007343, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001234, 0.0001274
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005004, 0.0004893
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011870, 0.0011886
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012864, 0.0013337
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006398, 0.0006138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002995, upper bound: 0.0002999
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002995, upper bound: 0.0003001
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006562, 0.0006396
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002245, 0.0002163
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008285, 0.0008036
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007325, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001230, 0.0001277
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0005012, 0.0004881
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011868, 0.0011887
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012816, 0.0013367
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006414, 0.0006112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002350, upper bound: 0.0002349
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002350, upper bound: 0.0002349
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006368, 0.0006243
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002145, 0.0002081
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007984, 0.0007799
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007142, 0.0007302
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001186, 0.0001221
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004859, 0.0004761
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011850, 0.0011865
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012299, 0.0012719
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006071, 0.0005836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003223, upper bound: 0.0003590
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003432, upper bound: 0.0003220
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006399, 0.0006199
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002159, 0.0002060
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008033, 0.0007732
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007088, 0.0007337
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001174, 0.0001230
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004883, 0.0004726
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011845, 0.0011868
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012153, 0.0012821
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006122, 0.0005758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002829, upper bound: 0.0002885
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002814, upper bound: 0.0002892
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006276, 0.0005959
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002175, 0.0002018
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007817, 0.0007328
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0006997, 0.0007395
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001094, 0.0001185
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004785, 0.0004535
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011820, 0.0011857
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011276, 0.0012334
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006003, 0.0005437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002726, upper bound: 0.0002728
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002726, upper bound: 0.0002728
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006016, 0.0006190
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002048, 0.0002131
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007417, 0.0007684
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007277, 0.0007079
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001161, 0.0001111
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004581, 0.0004717
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011847, 0.0011827
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012048, 0.0011468
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005540, 0.0005850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003512, upper bound: 0.0003196
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003432, upper bound: 0.0003220
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006353, 0.0006373
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002103, 0.0002105
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007950, 0.0007980
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007218, 0.0007219
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001211, 0.0001205
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004848, 0.0004863
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011867, 0.0011865
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012539, 0.0012474
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005856, 0.0005891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002857, upper bound: 0.0002862
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002843, upper bound: 0.0002864
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006416, 0.0006315
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002134, 0.0002077
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008047, 0.0007892
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007149, 0.0007296
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001195, 0.0001224
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004897, 0.0004818
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011861, 0.0011872
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012348, 0.0012684
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005969, 0.0005789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003175, upper bound: 0.0003157
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003155, upper bound: 0.0003192
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006534, 0.0006464
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002232, 0.0002194
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008241, 0.0008140
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007398, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001250, 0.0001269
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004989, 0.0004934
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011876, 0.0011884
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013041, 0.0013276
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006365, 0.0006233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002846, upper bound: 0.0002829
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002840, upper bound: 0.0002844
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006544, 0.0006449
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002236, 0.0002187
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008257, 0.0008118
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007381, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001245, 0.0001271
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004998, 0.0004923
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011874, 0.0011885
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012993, 0.0013306
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006381, 0.0006207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002350, upper bound: 0.0002349
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002350, upper bound: 0.0002349
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006189, 0.0006311
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002066, 0.0002118
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007708, 0.0007905
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007211, 0.0007087
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001206, 0.0001169
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004718, 0.0004815
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011858, 0.0011844
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012527, 0.0012122
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005755, 0.0005958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002349, upper bound: 0.0002348
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002349, upper bound: 0.0002348
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006395, 0.0006460
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002165, 0.0002191
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008027, 0.0008133
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007392, 0.0007333
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001248, 0.0001228
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004880, 0.0004932
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011875, 0.0011868
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013023, 0.0012807
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006115, 0.0006222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002827, upper bound: 0.0002891
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002814, upper bound: 0.0002892
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006387, 0.0006493
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002158, 0.0002210
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008023, 0.0008186
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007444, 0.0007314
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001258, 0.0001228
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004874, 0.0004958
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011879, 0.0011867
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013140, 0.0012787
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006097, 0.0006286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002797, upper bound: 0.0002791
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002797, upper bound: 0.0002791
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006398, 0.0006480
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002163, 0.0002203
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008038, 0.0008165
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007427, 0.0007325
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001254, 0.0001230
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004883, 0.0004947
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011878, 0.0011868
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013095, 0.0012816
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006112, 0.0006261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002833, upper bound: 0.0002832
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002829, upper bound: 0.0002846
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006010, 0.0006341
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001974, 0.0002136
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007441, 0.0007950
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007262, 0.0006859
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001214, 0.0001119
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004577, 0.0004838
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011861, 0.0011823
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012626, 0.0011524
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005421, 0.0006010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002736, upper bound: 0.0002713
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002736, upper bound: 0.0002713
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006248, 0.0006490
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002091, 0.0002208
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007808, 0.0008180
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007438, 0.0007149
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001257, 0.0001188
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004765, 0.0004956
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011879, 0.0011851
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013124, 0.0012318
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005846, 0.0006276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002864, upper bound: 0.0002830
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002859, upper bound: 0.0002843
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006384, 0.0006499
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002157, 0.0002213
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008017, 0.0008195
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007451, 0.0007310
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001260, 0.0001227
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004872, 0.0004962
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011880, 0.0011866
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013159, 0.0012775
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006090, 0.0006296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002787, upper bound: 0.0002765
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002778, upper bound: 0.0002775
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006394, 0.0006484
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002161, 0.0002206
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008032, 0.0008172
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007433, 0.0007321
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001255, 0.0001229
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004880, 0.0004951
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011878, 0.0011868
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013110, 0.0012804
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006106, 0.0006269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003001, upper bound: 0.0002991
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003005, upper bound: 0.0002978
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006018, 0.0006285
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001965, 0.0002096
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007454, 0.0007864
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007165, 0.0006840
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001198, 0.0001122
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004584, 0.0004794
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011855, 0.0011824
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012440, 0.0011552
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005436, 0.0005911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002790, upper bound: 0.0002764
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002790, upper bound: 0.0002764
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006184, 0.0006495
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002047, 0.0002210
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007710, 0.0008187
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007444, 0.0007042
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001258, 0.0001169
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004715, 0.0004959
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011879, 0.0011843
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013139, 0.0012105
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005732, 0.0006285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003171, upper bound: 0.0003157
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003156, upper bound: 0.0003175
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006210, 0.0006002
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002147, 0.0002042
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007714, 0.0007396
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007062, 0.0007323
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001107, 0.0001165
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004733, 0.0004570
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011825, 0.0011849
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011422, 0.0012143
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005899, 0.0005515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002713, upper bound: 0.0002641
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002653, upper bound: 0.0002665
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0005942, 0.0006267
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002014, 0.0002171
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007301, 0.0007802
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007384, 0.0006992
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001183, 0.0001089
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004522, 0.0004778
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011856, 0.0011818
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012304, 0.0011233
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005410, 0.0005986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 47

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002495, upper bound: 0.0002436
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002495, upper bound: 0.0002436
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006227, 0.0006339
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002077, 0.0002128
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007768, 0.0007948
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007259, 0.0007131
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001214, 0.0001180
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004748, 0.0004837
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011861, 0.0011848
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012620, 0.0012251
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005821, 0.0006007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003302, upper bound: 0.0003422
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003224, upper bound: 0.0003470
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006266, 0.0006313
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002094, 0.0002115
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007829, 0.0007907
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007227, 0.0007175
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001206, 0.0001192
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004778, 0.0004816
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011858, 0.0011853
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012533, 0.0012378
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005886, 0.0005961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003306, upper bound: 0.0003423
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003261, upper bound: 0.0003475
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006419, 0.0006507
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002176, 0.0002217
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008065, 0.0008207
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007359
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001262, 0.0001236
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004899, 0.0004969
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011881, 0.0011871
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013187, 0.0012895
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006161, 0.0006311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002532, upper bound: 0.0002506
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002532, upper bound: 0.0002506
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006430, 0.0006492
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002180, 0.0002210
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008081, 0.0008184
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007442, 0.0007369
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001258, 0.0001238
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004907, 0.0004957
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011879, 0.0011872
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013137, 0.0012924
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006177, 0.0006284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002725, upper bound: 0.0002727
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002730, upper bound: 0.0002721
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006135, 0.0006047
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002106, 0.0002062
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007599, 0.0007465
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007105, 0.0007223
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001120, 0.0001145
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004674, 0.0004605
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011830, 0.0011841
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011572, 0.0011864
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005751, 0.0005595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002725, upper bound: 0.0002728
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002730, upper bound: 0.0002721
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0005905, 0.0006318
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001994, 0.0002194
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007246, 0.0007881
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007433, 0.0006943
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001197, 0.0001079
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004493, 0.0004818
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011862, 0.0011814
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012474, 0.0011097
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005342, 0.0006077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002797, upper bound: 0.0002791
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002792, upper bound: 0.0002795
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006194, 0.0006376
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002057, 0.0002146
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007724, 0.0008005
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007298, 0.0007082
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001224, 0.0001172
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004722, 0.0004866
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011866, 0.0011844
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012744, 0.0012136
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005748, 0.0006073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002780, upper bound: 0.0002770
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002770, upper bound: 0.0002783
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006234, 0.0006356
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002076, 0.0002136
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007785, 0.0007973
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007273, 0.0007130
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001218, 0.0001183
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004754, 0.0004850
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011863, 0.0011849
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012675, 0.0012269
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005819, 0.0006037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002988, upper bound: 0.0003002
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002996, upper bound: 0.0002995
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006378, 0.0006552
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002154, 0.0002237
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008008, 0.0008276
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007303
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001275, 0.0001225
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004867, 0.0005004
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011886, 0.0011866
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013336, 0.0012755
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006080, 0.0006390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002787, upper bound: 0.0002765
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002790, upper bound: 0.0002764
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006389, 0.0006537
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002158, 0.0002230
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008023, 0.0008254
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007314
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001271, 0.0001228
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004875, 0.0004993
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011884, 0.0011867
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013287, 0.0012785
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006095, 0.0006364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002532, upper bound: 0.0002506
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002532, upper bound: 0.0002506
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006241, 0.0006466
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002048, 0.0002151
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007777, 0.0008124
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007331, 0.0007082
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001238, 0.0001173
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004759, 0.0004936
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011878, 0.0011852
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012850, 0.0012099
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005656, 0.0006057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002718
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002720
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006296, 0.0006396
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002075, 0.0002117
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007862, 0.0008016
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007247, 0.0007149
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001218, 0.0001189
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004802, 0.0004881
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011870, 0.0011858
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012616, 0.0012283
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005754, 0.0005933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003013, upper bound: 0.0003020
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003031, upper bound: 0.0002998
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006063, 0.0006341
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001991, 0.0002118
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007514, 0.0007950
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007214, 0.0006904
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001214, 0.0001133
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004619, 0.0004838
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011861, 0.0011829
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012626, 0.0011702
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005531, 0.0006010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002833, upper bound: 0.0002832
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002846, upper bound: 0.0002828
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006223, 0.0006548
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002068, 0.0002234
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007763, 0.0008269
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007095
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001274, 0.0001179
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004745, 0.0005001
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011886, 0.0011848
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013317, 0.0012235
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005809, 0.0006380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002535, upper bound: 0.0002506
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002535, upper bound: 0.0002506
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006063, 0.0006341
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001991, 0.0002118
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007514, 0.0007950
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007214, 0.0006904
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001214, 0.0001133
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004619, 0.0004838
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011861, 0.0011829
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012626, 0.0011702
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005531, 0.0006010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002780, upper bound: 0.0002770
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002783, upper bound: 0.0002769
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006223, 0.0006548
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002068, 0.0002234
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007763, 0.0008269
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007095
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001274, 0.0001179
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004745, 0.0005001
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011886, 0.0011848
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013317, 0.0012235
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005809, 0.0006380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002192, upper bound: 0.0002179
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002192, upper bound: 0.0002179
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006066, 0.0006349
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001993, 0.0002122
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007519, 0.0007962
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007224, 0.0006909
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001216, 0.0001134
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004621, 0.0004844
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011862, 0.0011829
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012652, 0.0011714
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005537, 0.0006024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002671, upper bound: 0.0002628
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002670, upper bound: 0.0002636
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006214, 0.0006556
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002063, 0.0002238
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007749, 0.0008281
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007084
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001276, 0.0001177
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004738, 0.0005007
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011887, 0.0011847
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013344, 0.0012205
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005793, 0.0006394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002760, upper bound: 0.0002718
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002721, upper bound: 0.0002725
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006184, 0.0006060
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002134, 0.0002068
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007674, 0.0007484
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007120, 0.0007292
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001123, 0.0001158
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004713, 0.0004615
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011832, 0.0011846
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011614, 0.0012056
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005852, 0.0005618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002495, upper bound: 0.0002438
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002495, upper bound: 0.0002438
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0005932, 0.0006321
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002009, 0.0002195
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007287, 0.0007885
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007436, 0.0006981
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001198, 0.0001086
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004514, 0.0004820
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011862, 0.0011817
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012483, 0.0011202
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005394, 0.0006082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002715, upper bound: 0.0002641
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002653, upper bound: 0.0002665
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006329, 0.0006225
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002123, 0.0002072
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007932, 0.0007773
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007120, 0.0007247
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001181, 0.0001211
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004829, 0.0004747
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011848, 0.0011860
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012241, 0.0012587
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005989, 0.0005805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003175, upper bound: 0.0003156
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003157, upper bound: 0.0003171
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006343, 0.0006186
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002130, 0.0002053
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007954, 0.0007712
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007072, 0.0007264
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001170, 0.0001215
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004840, 0.0004716
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011843, 0.0011862
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012110, 0.0012634
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006014, 0.0005735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002636, upper bound: 0.0002671
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002627, upper bound: 0.0002671
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006127, 0.0006240
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002032, 0.0002087
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007622, 0.0007795
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007138, 0.0007002
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001185, 0.0001153
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004670, 0.0004758
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011850, 0.0011836
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012289, 0.0011914
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005630, 0.0005830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 100

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002187, upper bound: 0.0002192
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002187, upper bound: 0.0002192
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006357, 0.0006389
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002144, 0.0002158
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007976, 0.0008025
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007315, 0.0007282
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001228, 0.0001219
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004851, 0.0004876
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011867, 0.0011863
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012787, 0.0012681
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006040, 0.0006096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002636, upper bound: 0.0002670
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002628, upper bound: 0.0002671
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006532, 0.0006392
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002231, 0.0002161
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008239, 0.0008031
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007321, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001229, 0.0001268
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004988, 0.0004878
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011867, 0.0011884
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012804, 0.0013270
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006362, 0.0006106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002774, upper bound: 0.0002778
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002775, upper bound: 0.0002778
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006542, 0.0006378
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002235, 0.0002154
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008254, 0.0008008
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007303, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001225, 0.0001271
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004996, 0.0004867
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011866, 0.0011885
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012755, 0.0013299
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006378, 0.0006080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002774, upper bound: 0.0002778
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002775, upper bound: 0.0002778
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006352, 0.0005896
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002216, 0.0001990
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007932, 0.0007232
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0006932, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001076, 0.0001206
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004845, 0.0004486
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011812, 0.0011866
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011067, 0.0012615
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006151, 0.0005326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002926, upper bound: 0.0002993
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002931, upper bound: 0.0002985
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006051, 0.0006129
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002067, 0.0002104
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007469, 0.0007591
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007217, 0.0007125
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001143, 0.0001120
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004608, 0.0004670
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011840, 0.0011831
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011846, 0.0011596
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005605, 0.0005741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 100

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002945, upper bound: 0.0002986
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002946, upper bound: 0.0002981
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006358, 0.0006300
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002106, 0.0002077
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007958, 0.0007869
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007154, 0.0007225
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001190, 0.0001207
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004852, 0.0004806
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011859, 0.0011866
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012298, 0.0012490
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005865, 0.0005762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003191, upper bound: 0.0003151
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003191, upper bound: 0.0003145
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006419, 0.0006250
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002135, 0.0002053
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008051, 0.0007792
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007093, 0.0007299
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001176, 0.0001224
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004899, 0.0004767
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011853, 0.0011873
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012130, 0.0012692
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005973, 0.0005673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003167, upper bound: 0.0003166
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003175, upper bound: 0.0003165
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006358, 0.0006300
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002106, 0.0002077
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007958, 0.0007869
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007154, 0.0007225
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001190, 0.0001207
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004852, 0.0004806
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011859, 0.0011866
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012298, 0.0012490
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005865, 0.0005762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002719, upper bound: 0.0002736
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002713, upper bound: 0.0002736
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006419, 0.0006250
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002135, 0.0002053
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008051, 0.0007792
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007093, 0.0007299
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001176, 0.0001224
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004899, 0.0004767
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011853, 0.0011873
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012130, 0.0012692
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005973, 0.0005673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002832, upper bound: 0.0002833
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002832, upper bound: 0.0002833
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006143, 0.0006188
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002031, 0.0002048
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007638, 0.0007715
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007046, 0.0007003
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001170, 0.0001156
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004682, 0.0004717
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011843, 0.0011838
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012115, 0.0011971
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005674, 0.0005737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002781, upper bound: 0.0002771
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002781, upper bound: 0.0002771
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006337, 0.0006398
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002123, 0.0002162
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007938, 0.0008037
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007325, 0.0007233
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001230, 0.0001212
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004834, 0.0004882
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011868, 0.0011861
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012815, 0.0012614
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006012, 0.0006111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002632, upper bound: 0.0002676
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002627, upper bound: 0.0002676
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006343, 0.0005905
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002211, 0.0001994
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007917, 0.0007246
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0006943, 0.0007460
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001079, 0.0001203
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004837, 0.0004493
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011814, 0.0011865
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011097, 0.0012584
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006135, 0.0005342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002436, upper bound: 0.0002495
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002436, upper bound: 0.0002495
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006055, 0.0006135
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002069, 0.0002106
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007475, 0.0007599
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007223, 0.0007130
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001145, 0.0001121
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004611, 0.0004674
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011841, 0.0011831
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011864, 0.0011611
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005612, 0.0005751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002948, upper bound: 0.0002967
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002948, upper bound: 0.0002967
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006009, 0.0006295
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001961, 0.0002101
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007441, 0.0007880
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007177, 0.0006829
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001201, 0.0001119
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004577, 0.0004802
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011856, 0.0011823
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012474, 0.0011522
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005420, 0.0005929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002778, upper bound: 0.0002777
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002778, upper bound: 0.0002777
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006184, 0.0006505
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002046, 0.0002215
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007709, 0.0008203
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007456, 0.0007042
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001261, 0.0001169
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004714, 0.0004967
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011881, 0.0011843
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013173, 0.0012103
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005731, 0.0006303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002661, upper bound: 0.0002643
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002661, upper bound: 0.0002649
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006150, 0.0006013
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002114, 0.0002047
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007622, 0.0007411
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007074, 0.0007242
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001110, 0.0001149
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004686, 0.0004578
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011826, 0.0011842
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011456, 0.0011914
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005778, 0.0005533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003220, upper bound: 0.0003432
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003194, upper bound: 0.0003512
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0005901, 0.0006269
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001992, 0.0002172
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007240, 0.0007806
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007387, 0.0006939
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001183, 0.0001078
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004490, 0.0004780
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011856, 0.0011813
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012312, 0.0011085
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005335, 0.0005991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002474, upper bound: 0.0002461
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002474, upper bound: 0.0002461
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006236, 0.0006406
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002046, 0.0002129
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007769, 0.0008031
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007283, 0.0007076
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001221, 0.0001172
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004755, 0.0004889
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011871, 0.0011851
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012649, 0.0012082
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005647, 0.0005950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002886, upper bound: 0.0002829
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002885, upper bound: 0.0002844
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006289, 0.0006343
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002072, 0.0002098
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007852, 0.0007934
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007206, 0.0007141
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001203, 0.0001187
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004797, 0.0004839
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011864, 0.0011858
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012439, 0.0012260
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005742, 0.0005838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002647, upper bound: 0.0002695
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002647, upper bound: 0.0002695
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006129, 0.0006006
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002103, 0.0002043
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007590, 0.0007401
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007066, 0.0007216
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001108, 0.0001143
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004669, 0.0004572
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011825, 0.0011840
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011434, 0.0011843
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005740, 0.0005522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002474, upper bound: 0.0002463
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002474, upper bound: 0.0002463
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0005889, 0.0006267
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0001986, 0.0002171
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007222, 0.0007803
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007385, 0.0006924
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001183, 0.0001074
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004481, 0.0004778
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011856, 0.0011812
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012305, 0.0011045
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005314, 0.0005987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002687, upper bound: 0.0002653
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002647, upper bound: 0.0002695
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006418, 0.0006513
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002176, 0.0002220
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008065, 0.0008216
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007460, 0.0007358
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001264, 0.0001236
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004899, 0.0004973
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011882, 0.0011871
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013205, 0.0012893
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006161, 0.0006320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Candidate
type: RSZ, layer: 3, pos: 100

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002830, upper bound: 0.0002840
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002821, upper bound: 0.0002851
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006429, 0.0006498
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002180, 0.0002213
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0008080, 0.0008193
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007450, 0.0007369
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001259, 0.0001238
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004907, 0.0004962
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011880, 0.0011872
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0013156, 0.0012922
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0006176, 0.0006294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002987, upper bound: 0.0003002
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002987, upper bound: 0.0003002
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006242, 0.0006345
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002084, 0.0002131
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007791, 0.0007957
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007266, 0.0007149
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001215, 0.0001184
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004760, 0.0004841
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011862, 0.0011850
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012640, 0.0012299
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005847, 0.0006018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003179, upper bound: 0.0003511
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0003486, upper bound: 0.0003298
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006265, 0.0006317
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002094, 0.0002117
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007828, 0.0007913
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007231, 0.0007174
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001207, 0.0001191
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004778, 0.0004819
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011859, 0.0011853
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012545, 0.0012376
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005885, 0.0005967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002999, upper bound: 0.0002995
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0002999, upper bound: 0.0002995
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0006183, 0.0006009
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002133, 0.0002045
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007672, 0.0007406
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007070, 0.0007291
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001109, 0.0001157
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004712, 0.0004575
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011826, 0.0011846
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0011445, 0.0012053
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005851, 0.0005528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 100
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002474, upper bound: 0.0002463
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0002474, upper bound: 0.0002463
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0013065, -0.0003996, -0.0013065, -0.0003996, -0.0005925, 0.0006270
1: -0.0042621, -0.0039711, -0.0042621, -0.0039711, -0.0002006, 0.0002173
2: 0.0128519, 0.0140634, 0.0128519, 0.0140634, -0.0007276, 0.0007807
3: 1.0083727, 1.0091187, 1.0083727, 1.0091187, -0.0007388, 0.0006973
4: -0.0038859, -0.0036880, -0.0038859, -0.0036880, -0.0001183, 0.0001084
5: 0.0029452, 0.0036436, 0.0029452, 0.0036436, -0.0004509, 0.0004780
6: -0.0024452, -0.0023745, -0.0024452, -0.0023745, -0.0000707, 0.0000707
7: -0.0129490, -0.0117312, -0.0129490, -0.0117312, -0.0011856, 0.0011816
8: -0.0094433, -0.0073017, -0.0094433, -0.0073017, -0.0012315, 0.0011179
9: -0.0007167, 0.0003426, -0.0007167, 0.0003426, -0.0005382, 0.0005992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.99 + 598.03 = 601.02 seconds
