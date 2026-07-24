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
execution time: IAR + RelationalAnalysis = 1.15 + 2.03 = 3.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0117933, upper bound: 0.0117933

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0116533, upper bound: 0.0115087
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0115087, upper bound: 0.0116533
time: 1.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.44
Output dim: 7, lower bound: -0.0116533, upper bound: 0.0115087
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.44
Output dim: 7, lower bound: -0.0115087, upper bound: 0.0116533

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113269, 0.0113603
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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
time: 0.97 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113603, 0.0113269
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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907
time: 0.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112785, 0.0113267
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
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108165
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113269, 0.0113119
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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108166
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113119, 0.0112953
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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108166, upper bound: 0.0110572
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0113603, 0.0112785
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
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108165, upper bound: 0.0110572
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108165
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108166
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 7, lower bound: -0.0108166, upper bound: 0.0110572
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 7, lower bound: -0.0108165, upper bound: 0.0110572
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108507, upper bound: 0.0110297
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108860, upper bound: 0.0110225
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110521, upper bound: 0.0108067
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110550, upper bound: 0.0107805
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108507, upper bound: 0.0110297
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108860, upper bound: 0.0110225
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110521, upper bound: 0.0108068
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110550, upper bound: 0.0107835
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107835, upper bound: 0.0110550
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0110521
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110225, upper bound: 0.0108860
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110297, upper bound: 0.0108507
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107805, upper bound: 0.0110550
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108067, upper bound: 0.0110521
time: 1.20 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110225, upper bound: 0.0108860
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110297, upper bound: 0.0108507
time: 1.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0108507, upper bound: 0.0110297
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0108860, upper bound: 0.0110225
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0110521, upper bound: 0.0108067
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0110550, upper bound: 0.0107805
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0108507, upper bound: 0.0110297
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0108860, upper bound: 0.0110225
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0110521, upper bound: 0.0108068
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0110550, upper bound: 0.0107835
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0107835, upper bound: 0.0110550
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0110521
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0110225, upper bound: 0.0108860
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0110297, upper bound: 0.0108507
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0107805, upper bound: 0.0110550
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0108067, upper bound: 0.0110521
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0110225, upper bound: 0.0108860
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 7, lower bound: -0.0110297, upper bound: 0.0108507

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107831, upper bound: 0.0102942
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103206, upper bound: 0.0109855
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108247, upper bound: 0.0104156
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102620, upper bound: 0.0109731
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110262, upper bound: 0.0102497
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104475, upper bound: 0.0107295
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110318, upper bound: 0.0102968
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103047, upper bound: 0.0106974
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107830, upper bound: 0.0103046
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103070, upper bound: 0.0109855
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108247, upper bound: 0.0104394
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102541, upper bound: 0.0109735
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110262, upper bound: 0.0102589
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104243, upper bound: 0.0107307
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110318, upper bound: 0.0103117
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102945, upper bound: 0.0107029
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107029, upper bound: 0.0102945
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103117, upper bound: 0.0110318
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107307, upper bound: 0.0104243
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102589, upper bound: 0.0110262
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109735, upper bound: 0.0102541
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104394, upper bound: 0.0108247
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109855, upper bound: 0.0103070
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103046, upper bound: 0.0107830
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106974, upper bound: 0.0103047
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102968, upper bound: 0.0110318
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107295, upper bound: 0.0104475
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102497, upper bound: 0.0110262
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112102, 0.0111997
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
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109731, upper bound: 0.0102620
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104156, upper bound: 0.0108247
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112022, 0.0112049
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
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109855, upper bound: 0.0103206
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102942, upper bound: 0.0107831
time: 1.13 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0107831, upper bound: 0.0102942
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0103206, upper bound: 0.0109855
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0108247, upper bound: 0.0104156
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0102620, upper bound: 0.0109731
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0110262, upper bound: 0.0102497
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0104475, upper bound: 0.0107295
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0110318, upper bound: 0.0102968
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0103047, upper bound: 0.0106974
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0107830, upper bound: 0.0103046
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0103070, upper bound: 0.0109855
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0108247, upper bound: 0.0104394
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0102541, upper bound: 0.0109735
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0110262, upper bound: 0.0102589
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0104243, upper bound: 0.0107307
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0110318, upper bound: 0.0103117
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0102945, upper bound: 0.0107029
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0107029, upper bound: 0.0102945
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0103117, upper bound: 0.0110318
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0107307, upper bound: 0.0104243
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0102589, upper bound: 0.0110262
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0109735, upper bound: 0.0102541
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0104394, upper bound: 0.0108247
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0109855, upper bound: 0.0103070
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0103046, upper bound: 0.0107830
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0106974, upper bound: 0.0103047
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0102968, upper bound: 0.0110318
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0107295, upper bound: 0.0104475
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0102497, upper bound: 0.0110262
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0109731, upper bound: 0.0102620
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0104156, upper bound: 0.0108247
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0109855, upper bound: 0.0103206
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 7, lower bound: -0.0102942, upper bound: 0.0107831

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0108894, 0.0109644
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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107654, upper bound: 0.0100690
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106477, upper bound: 0.0102760
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103032, upper bound: 0.0107272
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101914, upper bound: 0.0109667
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0108894, 0.0109644
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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0102228
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106638, upper bound: 0.0103975
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102442, upper bound: 0.0107257
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100730, upper bound: 0.0109544
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110076, upper bound: 0.0100571
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108316, upper bound: 0.0102317
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104295, upper bound: 0.0105562
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102804, upper bound: 0.0107117
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110132, upper bound: 0.0101540
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108329, upper bound: 0.0102793
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102868, upper bound: 0.0105424
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100907, upper bound: 0.0106799
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107654, upper bound: 0.0100897
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106458, upper bound: 0.0102867
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102894, upper bound: 0.0107489
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101659, upper bound: 0.0109667
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0102694
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106635, upper bound: 0.0104214
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102361, upper bound: 0.0107487
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100603, upper bound: 0.0109548
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110076, upper bound: 0.0100701
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108302, upper bound: 0.0102410
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109558, 0.0109221
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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104062, upper bound: 0.0105685
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102343, upper bound: 0.0107130
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110132, upper bound: 0.0101798
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108314, upper bound: 0.0102942
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109558, 0.0109221
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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102763, upper bound: 0.0105572
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100706, upper bound: 0.0106855
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106855, upper bound: 0.0100706
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105572, upper bound: 0.0102763
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102942, upper bound: 0.0108314
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101798, upper bound: 0.0110132
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107130, upper bound: 0.0102343
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105685, upper bound: 0.0104062
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102410, upper bound: 0.0108302
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0110076
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109548, upper bound: 0.0100603
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107487, upper bound: 0.0102361
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104214, upper bound: 0.0106635
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102694, upper bound: 0.0108068
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109667, upper bound: 0.0101659
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107489, upper bound: 0.0102894
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102867, upper bound: 0.0106458
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100897, upper bound: 0.0107654
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109715, 0.0109072
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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106799, upper bound: 0.0100907
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105424, upper bound: 0.0102868
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102793, upper bound: 0.0108329
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101540, upper bound: 0.0110132
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109715, 0.0109072
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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107117, upper bound: 0.0102804
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105562, upper bound: 0.0104295
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102317, upper bound: 0.0108316
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0110076
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0108956, 0.0109931
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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109544, upper bound: 0.0100730
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107257, upper bound: 0.0102442
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109988, 0.0108894
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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103975, upper bound: 0.0106638
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102228, upper bound: 0.0108068
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0108956, 0.0109931
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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109667, upper bound: 0.0101914
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107272, upper bound: 0.0103032
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0109988, 0.0108894
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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 106

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102760, upper bound: 0.0106477
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0107654
time: 1.42 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0107654, upper bound: 0.0100690
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0106477, upper bound: 0.0102760
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0103032, upper bound: 0.0107272
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0101914, upper bound: 0.0109667
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0102228
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0106638, upper bound: 0.0103975
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102442, upper bound: 0.0107257
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0100730, upper bound: 0.0109544
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0110076, upper bound: 0.0100571
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0108316, upper bound: 0.0102317
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0104295, upper bound: 0.0105562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102804, upper bound: 0.0107117
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0110132, upper bound: 0.0101540
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0108329, upper bound: 0.0102793
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102868, upper bound: 0.0105424
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0100907, upper bound: 0.0106799
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0107654, upper bound: 0.0100897
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0106458, upper bound: 0.0102867
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102894, upper bound: 0.0107489
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0101659, upper bound: 0.0109667
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0102694
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0106635, upper bound: 0.0104214
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102361, upper bound: 0.0107487
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0100603, upper bound: 0.0109548
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0110076, upper bound: 0.0100701
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0108302, upper bound: 0.0102410
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0104062, upper bound: 0.0105685
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102343, upper bound: 0.0107130
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0110132, upper bound: 0.0101798
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0108314, upper bound: 0.0102942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102763, upper bound: 0.0105572
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0100706, upper bound: 0.0106855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0106855, upper bound: 0.0100706
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0105572, upper bound: 0.0102763
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102942, upper bound: 0.0108314
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0101798, upper bound: 0.0110132
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0107130, upper bound: 0.0102343
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0105685, upper bound: 0.0104062
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102410, upper bound: 0.0108302
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0110076
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0109548, upper bound: 0.0100603
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0107487, upper bound: 0.0102361
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0104214, upper bound: 0.0106635
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102694, upper bound: 0.0108068
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0109667, upper bound: 0.0101659
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0107489, upper bound: 0.0102894
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102867, upper bound: 0.0106458
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0100897, upper bound: 0.0107654
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0106799, upper bound: 0.0100907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0105424, upper bound: 0.0102868
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102793, upper bound: 0.0108329
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0101540, upper bound: 0.0110132
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0107117, upper bound: 0.0102804
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0105562, upper bound: 0.0104295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102317, upper bound: 0.0108316
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0110076
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0109544, upper bound: 0.0100730
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0107257, upper bound: 0.0102442
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0103975, upper bound: 0.0106638
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102228, upper bound: 0.0108068
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0109667, upper bound: 0.0101914
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0107272, upper bound: 0.0103032
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0102760, upper bound: 0.0106477
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.87
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0107654

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0098925
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106098, upper bound: 0.0098928
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101073
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104951, upper bound: 0.0101062
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101133, upper bound: 0.0105764
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101458, upper bound: 0.0105025
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099990, upper bound: 0.0108062
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100350, upper bound: 0.0107557
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106161, upper bound: 0.0100747
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106470, upper bound: 0.0100177
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0102463
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105069, upper bound: 0.0101908
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100734, upper bound: 0.0105751
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100744, upper bound: 0.0104749
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099000, upper bound: 0.0107909
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099000, upper bound: 0.0106917
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107732, upper bound: 0.0098819
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108426, upper bound: 0.0098821
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100637
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106737, upper bound: 0.0100623
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100593, upper bound: 0.0105548
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101249, upper bound: 0.0105190
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108142, upper bound: 0.0099974
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108530, upper bound: 0.0099732
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106197, upper bound: 0.0101230
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106745, upper bound: 0.0100963
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0105298
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0104322
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0099165
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106098, upper bound: 0.0099165
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101186
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104940, upper bound: 0.0101147
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101057, upper bound: 0.0105918
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101319, upper bound: 0.0105102
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099852, upper bound: 0.0108062
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100086, upper bound: 0.0107557
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106161, upper bound: 0.0101157
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106470, upper bound: 0.0100430
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0102655
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105066, upper bound: 0.0101983
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100662, upper bound: 0.0105910
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100664, upper bound: 0.0104838
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098840, upper bound: 0.0107909
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098832, upper bound: 0.0106917
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107732, upper bound: 0.0098981
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108426, upper bound: 0.0098980
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100721
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106736, upper bound: 0.0100701
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100325, upper bound: 0.0105558
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100856, upper bound: 0.0105190
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108142, upper bound: 0.0100251
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108530, upper bound: 0.0099869
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106197, upper bound: 0.0101369
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106742, upper bound: 0.0101035
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098941, upper bound: 0.0105327
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0104325
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104325, upper bound: 0.0098934
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105327, upper bound: 0.0098941
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101035, upper bound: 0.0106742
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101369, upper bound: 0.0106197
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099869, upper bound: 0.0108530
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100251, upper bound: 0.0108142
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105190, upper bound: 0.0100856
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105558, upper bound: 0.0100325
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0106736
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100721, upper bound: 0.0106032
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098980, upper bound: 0.0108426
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098981, upper bound: 0.0107732
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111894, 0.0112824
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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0098832
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107909, upper bound: 0.0098840
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112163, 0.0112523
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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104838, upper bound: 0.0100664
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105910, upper bound: 0.0100662
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111894, 0.0112824
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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101983, upper bound: 0.0105066
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102655, upper bound: 0.0104737
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112163, 0.0112523
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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100430, upper bound: 0.0106470
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101157, upper bound: 0.0106161
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111894, 0.0112824
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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107557, upper bound: 0.0100086
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108062, upper bound: 0.0099852
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112163, 0.0112523
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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105102, upper bound: 0.0101319
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105918, upper bound: 0.0101057
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111894, 0.0112824
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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101147, upper bound: 0.0104940
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101186, upper bound: 0.0104316
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112163, 0.0112523
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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099165, upper bound: 0.0106098
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099165, upper bound: 0.0105259
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104322, upper bound: 0.0099168
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105298, upper bound: 0.0099168
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100963, upper bound: 0.0106745
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101230, upper bound: 0.0106197
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099732, upper bound: 0.0108530
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099974, upper bound: 0.0108142
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105190, upper bound: 0.0101249
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105548, upper bound: 0.0100593
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100623, upper bound: 0.0106737
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100637, upper bound: 0.0106032
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098821, upper bound: 0.0108426
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0107732
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112388, 0.0112595
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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0099000
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107909, upper bound: 0.0099000
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112647, 0.0112279
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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104749, upper bound: 0.0100744
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105751, upper bound: 0.0100734
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112388, 0.0112595
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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101908, upper bound: 0.0105069
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102463, upper bound: 0.0104737
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112647, 0.0112279
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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100177, upper bound: 0.0106470
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100747, upper bound: 0.0106161
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112388, 0.0112595
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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107557, upper bound: 0.0100350
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108062, upper bound: 0.0099990
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112647, 0.0112279
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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105024, upper bound: 0.0101458
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105764, upper bound: 0.0101133
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112388, 0.0112595
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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101062, upper bound: 0.0104951
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101073, upper bound: 0.0104316
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0112647, 0.0112279
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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0106098
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098925, upper bound: 0.0105259
time: 1.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0098925
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106098, upper bound: 0.0098928
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101073
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0104951, upper bound: 0.0101062
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101133, upper bound: 0.0105764
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101458, upper bound: 0.0105025
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099990, upper bound: 0.0108062
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100350, upper bound: 0.0107557
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106161, upper bound: 0.0100747
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106470, upper bound: 0.0100177
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0102463
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105069, upper bound: 0.0101908
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100734, upper bound: 0.0105751
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100744, upper bound: 0.0104749
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099000, upper bound: 0.0107909
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099000, upper bound: 0.0106917
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0107732, upper bound: 0.0098819
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0108426, upper bound: 0.0098821
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100637
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106737, upper bound: 0.0100623
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100593, upper bound: 0.0105548
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101249, upper bound: 0.0105190
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0108142, upper bound: 0.0099974
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0108530, upper bound: 0.0099732
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106197, upper bound: 0.0101230
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106745, upper bound: 0.0100963
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0105298
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0104322
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0099165
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106098, upper bound: 0.0099165
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101186
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0104940, upper bound: 0.0101147
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101057, upper bound: 0.0105918
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101319, upper bound: 0.0105102
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099852, upper bound: 0.0108062
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100086, upper bound: 0.0107557
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106161, upper bound: 0.0101157
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106470, upper bound: 0.0100430
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0102655
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105066, upper bound: 0.0101983
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100662, upper bound: 0.0105910
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100664, upper bound: 0.0104838
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0098840, upper bound: 0.0107909
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0098832, upper bound: 0.0106917
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0107732, upper bound: 0.0098981
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0108426, upper bound: 0.0098980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100721
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106736, upper bound: 0.0100701
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100325, upper bound: 0.0105558
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100856, upper bound: 0.0105190
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0108142, upper bound: 0.0100251
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0108530, upper bound: 0.0099869
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106197, upper bound: 0.0101369
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106742, upper bound: 0.0101035
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0098941, upper bound: 0.0105327
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0104325
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0104325, upper bound: 0.0098934
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105327, upper bound: 0.0098941
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101035, upper bound: 0.0106742
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101369, upper bound: 0.0106197
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099869, upper bound: 0.0108530
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100251, upper bound: 0.0108142
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105190, upper bound: 0.0100856
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105558, upper bound: 0.0100325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0106736
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100721, upper bound: 0.0106032
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0098980, upper bound: 0.0108426
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0098981, upper bound: 0.0107732
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0098832
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0107909, upper bound: 0.0098840
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0104838, upper bound: 0.0100664
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105910, upper bound: 0.0100662
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101983, upper bound: 0.0105066
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0102655, upper bound: 0.0104737
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100430, upper bound: 0.0106470
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101157, upper bound: 0.0106161
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0107557, upper bound: 0.0100086
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0108062, upper bound: 0.0099852
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105102, upper bound: 0.0101319
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105918, upper bound: 0.0101057
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101147, upper bound: 0.0104940
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101186, upper bound: 0.0104316
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099165, upper bound: 0.0106098
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099165, upper bound: 0.0105259
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0104322, upper bound: 0.0099168
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105298, upper bound: 0.0099168
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100963, upper bound: 0.0106745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101230, upper bound: 0.0106197
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099732, upper bound: 0.0108530
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0099974, upper bound: 0.0108142
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105190, upper bound: 0.0101249
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105548, upper bound: 0.0100593
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100623, upper bound: 0.0106737
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100637, upper bound: 0.0106032
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0098821, upper bound: 0.0108426
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0107732
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0099000
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0107909, upper bound: 0.0099000
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0104749, upper bound: 0.0100744
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105751, upper bound: 0.0100734
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101908, upper bound: 0.0105069
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0102463, upper bound: 0.0104737
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100177, upper bound: 0.0106470
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0100747, upper bound: 0.0106161
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0107557, upper bound: 0.0100350
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0108062, upper bound: 0.0099990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105024, upper bound: 0.0101458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0105764, upper bound: 0.0101133
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101062, upper bound: 0.0104951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0101073, upper bound: 0.0104316
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0106098
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0098925, upper bound: 0.0105259

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 106
type: RSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 106

### Candidate
type: RSZ, layer: 3, pos: 241

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.36 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.84
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0099000
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 7, lower bound: -0.0107909, upper bound: 0.0099000
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 7, lower bound: -0.0100177, upper bound: 0.0106470
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 7, lower bound: -0.0100747, upper bound: 0.0106161
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 7, lower bound: -0.0107557, upper bound: 0.0100350
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.84
Output dim: 7, lower bound: -0.0108062, upper bound: 0.0099990

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.18 + 598.06 = 601.24 seconds
