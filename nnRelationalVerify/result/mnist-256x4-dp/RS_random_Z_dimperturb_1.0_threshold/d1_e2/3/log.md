## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01061397


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758)
1: (-0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403)
2: (0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284)
3: (-0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0114900, 0.0114900)
4: (-0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521)
5: (0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548)
6: (-0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312)
7: (0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624)
8: (-0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763)
9: (-0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.12 + 2.02 = 3.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0117933, upper bound: 0.0117933

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112996, upper bound: 0.0116051
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0116051, upper bound: 0.0112996
time: 1.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.11 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.11
Output dim: 7, lower bound: -0.0112996, upper bound: 0.0116051
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.11
Output dim: 7, lower bound: -0.0116051, upper bound: 0.0112996

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0114125, 0.0113358
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0111358, upper bound: 0.0113511
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110216, upper bound: 0.0114973
time: 1.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113358, 0.0114125
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0110216
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0113511, upper bound: 0.0111358
time: 1.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 7, lower bound: -0.0111358, upper bound: 0.0113511
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 7, lower bound: -0.0110216, upper bound: 0.0114973
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 7, lower bound: -0.0114973, upper bound: 0.0110216
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 7, lower bound: -0.0113511, upper bound: 0.0111358

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112494, 0.0112062
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112821, 0.0111728
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108166, upper bound: 0.0110572
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108165, upper bound: 0.0110572
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111728, 0.0112821
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108165
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108166
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112062, 0.0112494
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 7, lower bound: -0.0108166, upper bound: 0.0110572
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 7, lower bound: -0.0108165, upper bound: 0.0110572
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108165
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108166
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112010, 0.0111726
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107048, upper bound: 0.0108754
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107330, upper bound: 0.0108363
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112494, 0.0111578
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108341, upper bound: 0.0104492
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103183, upper bound: 0.0109924
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112337, 0.0111412
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107418, upper bound: 0.0104347
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103225, upper bound: 0.0110356
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112821, 0.0111244
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106256, upper bound: 0.0108975
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106604, upper bound: 0.0108722
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111244, 0.0112551
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110356, upper bound: 0.0103084
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104570, upper bound: 0.0107408
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111728, 0.0112337
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108722, upper bound: 0.0106604
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108975, upper bound: 0.0106256
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111578, 0.0112239
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109924, upper bound: 0.0103183
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104492, upper bound: 0.0108341
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112062, 0.0112010
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100521
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100521
time: 1.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0107048, upper bound: 0.0108754
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0107330, upper bound: 0.0108363
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0108341, upper bound: 0.0104492
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0103183, upper bound: 0.0109924
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0107418, upper bound: 0.0104347
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0103225, upper bound: 0.0110356
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0106256, upper bound: 0.0108975
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0106604, upper bound: 0.0108722
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0110356, upper bound: 0.0103084
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0104570, upper bound: 0.0107408
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0108722, upper bound: 0.0106604
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0108975, upper bound: 0.0106256
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0109924, upper bound: 0.0103183
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0104492, upper bound: 0.0108341
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100521
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100521

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111355, 0.0110850
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106869, upper bound: 0.0106603
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0105525, upper bound: 0.0108567
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111135, 0.0111118
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098116, upper bound: 0.0098582
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098116, upper bound: 0.0098582
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109388, 0.0109502
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107830, upper bound: 0.0103046
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108247, upper bound: 0.0104394
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0110417, 0.0108462
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101257, upper bound: 0.0108321
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101607, upper bound: 0.0107830
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109221, 0.0109233
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100039, upper bound: 0.0098557
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100039, upper bound: 0.0098557
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0110360, 0.0108296
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103052, upper bound: 0.0108395
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101904, upper bound: 0.0110170
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112233, 0.0110368
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106076, upper bound: 0.0107352
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104535, upper bound: 0.0108790
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111946, 0.0110556
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105837, upper bound: 0.0102358
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101521, upper bound: 0.0108385
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0108127, 0.0110566
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108385, upper bound: 0.0101521
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108754, upper bound: 0.0101257
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109072, 0.0109435
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104390, upper bound: 0.0105681
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102898, upper bound: 0.0107230
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111116, 0.0111462
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108536, upper bound: 0.0105039
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106852, upper bound: 0.0106427
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0110852, 0.0111673
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108790, upper bound: 0.0104558
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107352, upper bound: 0.0106076
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0108462, 0.0110155
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100924, upper bound: 0.0097940
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100924, upper bound: 0.0097940
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109502, 0.0109123
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102272, upper bound: 0.0106740
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102929, upper bound: 0.0106433
time: 1.12 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0106869, upper bound: 0.0106603
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0105525, upper bound: 0.0108567
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0098116, upper bound: 0.0098582
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0098116, upper bound: 0.0098582
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0107830, upper bound: 0.0103046
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0108247, upper bound: 0.0104394
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0101257, upper bound: 0.0108321
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0101607, upper bound: 0.0107830
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0100039, upper bound: 0.0098557
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0100039, upper bound: 0.0098557
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0103052, upper bound: 0.0108395
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0101904, upper bound: 0.0110170
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0106076, upper bound: 0.0107352
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0104535, upper bound: 0.0108790
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0105837, upper bound: 0.0102358
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0101521, upper bound: 0.0108385
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0108385, upper bound: 0.0101521
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0108754, upper bound: 0.0101257
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0104390, upper bound: 0.0105681
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0102898, upper bound: 0.0107230
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0108536, upper bound: 0.0105039
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0106852, upper bound: 0.0106427
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0108790, upper bound: 0.0104558
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0107352, upper bound: 0.0106076
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0100924, upper bound: 0.0097940
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0100924, upper bound: 0.0097940
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0102272, upper bound: 0.0106740
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.64
Output dim: 7, lower bound: -0.0102929, upper bound: 0.0106433

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112279, 0.0112311
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0105970, upper bound: 0.0106510
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106786, upper bound: 0.0106495
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112595, 0.0112063
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0105042, upper bound: 0.0108519
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0105431, upper bound: 0.0108395
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112534, 0.0111528
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107654, upper bound: 0.0100897
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106458, upper bound: 0.0102867
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112491, 0.0111617
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100278, upper bound: 0.0098394
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100277, upper bound: 0.0098394
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111914, 0.0110702
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101171, upper bound: 0.0106013
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099849, upper bound: 0.0108134
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111619, 0.0110870
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101495, upper bound: 0.0107745
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100844, upper bound: 0.0107103
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112610, 0.0111997
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101148, upper bound: 0.0106825
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101478, upper bound: 0.0106284
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112922, 0.0111742
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101798, upper bound: 0.0110132
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0110076
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113105, 0.0111829
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0105203, upper bound: 0.0107282
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0105979, upper bound: 0.0107277
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113406, 0.0111557
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0096933, upper bound: 0.0098403
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0096933, upper bound: 0.0098403
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0110846, 0.0108127
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101406, upper bound: 0.0108328
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100817, upper bound: 0.0107916
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0110556, 0.0111676
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107916, upper bound: 0.0100817
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108328, upper bound: 0.0101406
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0110368, 0.0111971
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108569, upper bound: 0.0099849
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106828, upper bound: 0.0101082
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111829, 0.0112834
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102804, upper bound: 0.0107117
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100907, upper bound: 0.0106799
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112051, 0.0112922
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0096933
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0096933
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112313, 0.0112610
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104034, upper bound: 0.0106328
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106775, upper bound: 0.0106116
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112051, 0.0112922
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0096933
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0096933
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112313, 0.0112610
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097991, upper bound: 0.0097718
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097991, upper bound: 0.0097718
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0110870, 0.0111363
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102164, upper bound: 0.0106646
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101329, upper bound: 0.0106270
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0110702, 0.0111666
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0096167, upper bound: 0.0097930
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0096167, upper bound: 0.0097930
time: 1.18 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0105970, upper bound: 0.0106510
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0106786, upper bound: 0.0106495
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0105042, upper bound: 0.0108519
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0105431, upper bound: 0.0108395
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0107654, upper bound: 0.0100897
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0106458, upper bound: 0.0102867
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0100278, upper bound: 0.0098394
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0100277, upper bound: 0.0098394
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0101171, upper bound: 0.0106013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0099849, upper bound: 0.0108134
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0101495, upper bound: 0.0107745
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0100844, upper bound: 0.0107103
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0101148, upper bound: 0.0106825
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0101478, upper bound: 0.0106284
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0101798, upper bound: 0.0110132
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0110076
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0105203, upper bound: 0.0107282
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0105979, upper bound: 0.0107277
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0096933, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0096933, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0101406, upper bound: 0.0108328
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0100817, upper bound: 0.0107916
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0107916, upper bound: 0.0100817
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0108328, upper bound: 0.0101406
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0108569, upper bound: 0.0099849
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0106828, upper bound: 0.0101082
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0102804, upper bound: 0.0107117
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0100907, upper bound: 0.0106799
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0096933
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0096933
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0104034, upper bound: 0.0106328
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0106775, upper bound: 0.0106116
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0096933
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0096933
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0097991, upper bound: 0.0097718
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0097991, upper bound: 0.0097718
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0102164, upper bound: 0.0106646
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0101329, upper bound: 0.0106270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0096167, upper bound: 0.0097930
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 7, lower bound: -0.0096167, upper bound: 0.0097930

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112049, 0.0111697
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0098925
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101133, upper bound: 0.0105764
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111997, 0.0111765
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097901, upper bound: 0.0097559
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097901, upper bound: 0.0097559
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112049, 0.0111697
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097131, upper bound: 0.0098394
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097125, upper bound: 0.0098394
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111997, 0.0111765
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0102463
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099000, upper bound: 0.0107909
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112774, 0.0112163
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0099165
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106098, upper bound: 0.0099165
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113079, 0.0111894
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101186
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104940, upper bound: 0.0101147
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113079, 0.0111894
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099852, upper bound: 0.0108062
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098840, upper bound: 0.0107909
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112534, 0.0111528
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101319, upper bound: 0.0105102
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100086, upper bound: 0.0107557
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112491, 0.0111617
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100664, upper bound: 0.0104838
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098832, upper bound: 0.0106917
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111673, 0.0110536
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101035, upper bound: 0.0106742
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0106736
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111462, 0.0110821
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095537, upper bound: 0.0097796
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095537, upper bound: 0.0097796
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112376, 0.0111368
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099869, upper bound: 0.0108530
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100251, upper bound: 0.0108142
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112340, 0.0111450
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098980, upper bound: 0.0108426
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098981, upper bound: 0.0107732
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112861, 0.0111182
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104322, upper bound: 0.0099168
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100963, upper bound: 0.0106745
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112834, 0.0111282
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105190, upper bound: 0.0101249
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100623, upper bound: 0.0106737
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112861, 0.0111182
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095634, upper bound: 0.0098585
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095634, upper bound: 0.0098585
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112834, 0.0111282
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100637, upper bound: 0.0106032
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0107732
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111282, 0.0112575
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107732, upper bound: 0.0098819
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100636
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111182, 0.0112590
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095634
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095634
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111557, 0.0113136
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108426, upper bound: 0.0098821
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108530, upper bound: 0.0099732
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111829, 0.0112834
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097796, upper bound: 0.0095534
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097796, upper bound: 0.0095534
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111282, 0.0112575
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100593, upper bound: 0.0105548
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101249, upper bound: 0.0105190
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111182, 0.0112590
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0105298
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0104322
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111768, 0.0112340
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100721
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100325, upper bound: 0.0105558
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111676, 0.0112376
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106197, upper bound: 0.0101369
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098941, upper bound: 0.0105327
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111617, 0.0112246
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0096076, upper bound: 0.0097889
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0096076, upper bound: 0.0097889
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111528, 0.0112278
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095366, upper bound: 0.0097731
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095366, upper bound: 0.0097731
time: 1.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0098925
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0101133, upper bound: 0.0105764
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0097901, upper bound: 0.0097559
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0097901, upper bound: 0.0097559
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0097131, upper bound: 0.0098394
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0097125, upper bound: 0.0098394
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0102463
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0099000, upper bound: 0.0107909
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0099165
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0106098, upper bound: 0.0099165
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101186
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0104940, upper bound: 0.0101147
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0099852, upper bound: 0.0108062
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0098840, upper bound: 0.0107909
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0101319, upper bound: 0.0105102
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0100086, upper bound: 0.0107557
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0100664, upper bound: 0.0104838
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0098832, upper bound: 0.0106917
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0101035, upper bound: 0.0106742
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0106736
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0095537, upper bound: 0.0097796
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0095537, upper bound: 0.0097796
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0099869, upper bound: 0.0108530
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0100251, upper bound: 0.0108142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0098980, upper bound: 0.0108426
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0098981, upper bound: 0.0107732
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0104322, upper bound: 0.0099168
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0100963, upper bound: 0.0106745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0105190, upper bound: 0.0101249
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0100623, upper bound: 0.0106737
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0095634, upper bound: 0.0098585
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0095634, upper bound: 0.0098585
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0100637, upper bound: 0.0106032
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0107732
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0107732, upper bound: 0.0098819
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100636
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095634
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095634
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0108426, upper bound: 0.0098821
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0108530, upper bound: 0.0099732
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0097796, upper bound: 0.0095534
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0097796, upper bound: 0.0095534
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0100593, upper bound: 0.0105548
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0101249, upper bound: 0.0105190
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0105298
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0104322
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100721
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0100325, upper bound: 0.0105558
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0106197, upper bound: 0.0101369
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0098941, upper bound: 0.0105327
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0096076, upper bound: 0.0097889
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0096076, upper bound: 0.0097889
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0095366, upper bound: 0.0097731
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 7, lower bound: -0.0095366, upper bound: 0.0097731

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109931, 0.0108610
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112534, 0.0111528
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112491, 0.0111617
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113079, 0.0111894
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113079, 0.0111894
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112376, 0.0111368
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112340, 0.0111450
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111673, 0.0110536
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111462, 0.0110821
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111673, 0.0110536
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111462, 0.0110821
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0110846, 0.0108127
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0110846, 0.0108127
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113406, 0.0111557
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111557, 0.0113136
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111282, 0.0112575
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111182, 0.0112590
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0108621, 0.0110360
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 183

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
time: 1.00 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.13 + 315.28 = 318.42 seconds
