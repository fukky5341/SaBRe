## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01061397


## IAR start

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
execution time: IAR + RelationalAnalysis = 1.20 + 1.93 = 3.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0117933, upper bound: 0.0117933

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0116533, upper bound: 0.0115087
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0115087, upper bound: 0.0116533
time: 1.02 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.26 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.26
Output dim: 7, lower bound: -0.0116533, upper bound: 0.0115087
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.26
Output dim: 7, lower bound: -0.0115087, upper bound: 0.0116533

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
time: 0.88 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907
time: 0.91 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.91 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108165
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108166
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108166, upper bound: 0.0110572
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108165, upper bound: 0.0110572
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.09 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.30 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108165
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108166
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0108166, upper bound: 0.0110572
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0108165, upper bound: 0.0110572
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108507, upper bound: 0.0110297
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108860, upper bound: 0.0110225
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110521, upper bound: 0.0108067
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110550, upper bound: 0.0107805
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108507, upper bound: 0.0110297
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108860, upper bound: 0.0110225
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110521, upper bound: 0.0108068
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110550, upper bound: 0.0107835
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107835, upper bound: 0.0110550
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0110521
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110225, upper bound: 0.0108860
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110297, upper bound: 0.0108507
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107805, upper bound: 0.0110550
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108067, upper bound: 0.0110521
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110225, upper bound: 0.0108860
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110297, upper bound: 0.0108507
time: 1.02 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0108507, upper bound: 0.0110297
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0108860, upper bound: 0.0110225
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0110521, upper bound: 0.0108067
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0110550, upper bound: 0.0107805
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0108507, upper bound: 0.0110297
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0108860, upper bound: 0.0110225
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0110521, upper bound: 0.0108068
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0110550, upper bound: 0.0107835
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0107835, upper bound: 0.0110550
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0110521
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0110225, upper bound: 0.0108860
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0110297, upper bound: 0.0108507
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0107805, upper bound: 0.0110550
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0108067, upper bound: 0.0110521
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0110225, upper bound: 0.0108860
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 7, lower bound: -0.0110297, upper bound: 0.0108507

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107831, upper bound: 0.0102942
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103206, upper bound: 0.0109855
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108247, upper bound: 0.0104156
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102620, upper bound: 0.0109731
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110262, upper bound: 0.0102497
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104475, upper bound: 0.0107295
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110318, upper bound: 0.0102968
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103047, upper bound: 0.0106974
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107830, upper bound: 0.0103046
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103070, upper bound: 0.0109855
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108247, upper bound: 0.0104394
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102541, upper bound: 0.0109735
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110262, upper bound: 0.0102589
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104243, upper bound: 0.0107307
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110318, upper bound: 0.0103117
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102945, upper bound: 0.0107029
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107029, upper bound: 0.0102945
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103117, upper bound: 0.0110318
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107307, upper bound: 0.0104243
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102589, upper bound: 0.0110262
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109735, upper bound: 0.0102541
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104394, upper bound: 0.0108247
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109855, upper bound: 0.0103070
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103046, upper bound: 0.0107830
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106974, upper bound: 0.0103047
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102968, upper bound: 0.0110318
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107295, upper bound: 0.0104475
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102497, upper bound: 0.0110262
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109731, upper bound: 0.0102620
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104156, upper bound: 0.0108247
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109855, upper bound: 0.0103206
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102942, upper bound: 0.0107831
time: 1.03 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0107831, upper bound: 0.0102942
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0103206, upper bound: 0.0109855
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0108247, upper bound: 0.0104156
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0102620, upper bound: 0.0109731
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0110262, upper bound: 0.0102497
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0104475, upper bound: 0.0107295
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0110318, upper bound: 0.0102968
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0103047, upper bound: 0.0106974
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0107830, upper bound: 0.0103046
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0103070, upper bound: 0.0109855
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0108247, upper bound: 0.0104394
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0102541, upper bound: 0.0109735
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0110262, upper bound: 0.0102589
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0104243, upper bound: 0.0107307
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0110318, upper bound: 0.0103117
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0102945, upper bound: 0.0107029
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0107029, upper bound: 0.0102945
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0103117, upper bound: 0.0110318
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0107307, upper bound: 0.0104243
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0102589, upper bound: 0.0110262
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0109735, upper bound: 0.0102541
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0104394, upper bound: 0.0108247
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0109855, upper bound: 0.0103070
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0103046, upper bound: 0.0107830
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0106974, upper bound: 0.0103047
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0102968, upper bound: 0.0110318
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0107295, upper bound: 0.0104475
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0102497, upper bound: 0.0110262
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0109731, upper bound: 0.0102620
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0104156, upper bound: 0.0108247
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0109855, upper bound: 0.0103206
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 7, lower bound: -0.0102942, upper bound: 0.0107831

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107654, upper bound: 0.0100690
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106477, upper bound: 0.0102760
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103032, upper bound: 0.0107272
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101914, upper bound: 0.0109667
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0102228
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106638, upper bound: 0.0103975
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102442, upper bound: 0.0107257
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100730, upper bound: 0.0109544
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110076, upper bound: 0.0100571
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108316, upper bound: 0.0102317
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104295, upper bound: 0.0105562
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102804, upper bound: 0.0107117
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110132, upper bound: 0.0101540
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108329, upper bound: 0.0102793
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102868, upper bound: 0.0105424
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100907, upper bound: 0.0106799
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107654, upper bound: 0.0100897
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106458, upper bound: 0.0102867
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102894, upper bound: 0.0107489
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101659, upper bound: 0.0109667
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0102694
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106635, upper bound: 0.0104214
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102361, upper bound: 0.0107487
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0109548
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110076, upper bound: 0.0100701
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108302, upper bound: 0.0102410
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104062, upper bound: 0.0105685
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102343, upper bound: 0.0107130
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110132, upper bound: 0.0101798
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108314, upper bound: 0.0102942
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102763, upper bound: 0.0105572
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0106855
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106855, upper bound: 0.0100706
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105572, upper bound: 0.0102763
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102942, upper bound: 0.0108314
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101798, upper bound: 0.0110132
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107130, upper bound: 0.0102343
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105685, upper bound: 0.0104062
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102410, upper bound: 0.0108302
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0110076
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109548, upper bound: 0.0100603
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107487, upper bound: 0.0102361
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104214, upper bound: 0.0106635
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102694, upper bound: 0.0108068
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109667, upper bound: 0.0101659
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107489, upper bound: 0.0102894
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102867, upper bound: 0.0106458
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100897, upper bound: 0.0107654
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106799, upper bound: 0.0100907
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105424, upper bound: 0.0102868
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102793, upper bound: 0.0108329
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101540, upper bound: 0.0110132
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107117, upper bound: 0.0102804
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105562, upper bound: 0.0104295
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102317, upper bound: 0.0108316
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0110076
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109544, upper bound: 0.0100730
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107257, upper bound: 0.0102442
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0103975, upper bound: 0.0106638
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102228, upper bound: 0.0108068
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0109667, upper bound: 0.0101914
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107272, upper bound: 0.0103032
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102760, upper bound: 0.0106477
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0107654
time: 1.34 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.82 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0107654, upper bound: 0.0100690
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0106477, upper bound: 0.0102760
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0103032, upper bound: 0.0107272
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0101914, upper bound: 0.0109667
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0102228
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0106638, upper bound: 0.0103975
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102442, upper bound: 0.0107257
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0100730, upper bound: 0.0109544
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0110076, upper bound: 0.0100571
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0108316, upper bound: 0.0102317
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0104295, upper bound: 0.0105562
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102804, upper bound: 0.0107117
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0110132, upper bound: 0.0101540
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0108329, upper bound: 0.0102793
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102868, upper bound: 0.0105424
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0100907, upper bound: 0.0106799
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0107654, upper bound: 0.0100897
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0106458, upper bound: 0.0102867
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102894, upper bound: 0.0107489
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0101659, upper bound: 0.0109667
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0108068, upper bound: 0.0102694
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0106635, upper bound: 0.0104214
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102361, upper bound: 0.0107487
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0109548
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0110076, upper bound: 0.0100701
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0108302, upper bound: 0.0102410
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0104062, upper bound: 0.0105685
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102343, upper bound: 0.0107130
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0110132, upper bound: 0.0101798
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0108314, upper bound: 0.0102942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102763, upper bound: 0.0105572
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0106855
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0106855, upper bound: 0.0100706
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0105572, upper bound: 0.0102763
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102942, upper bound: 0.0108314
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0101798, upper bound: 0.0110132
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0107130, upper bound: 0.0102343
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0105685, upper bound: 0.0104062
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102410, upper bound: 0.0108302
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0110076
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0109548, upper bound: 0.0100603
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0107487, upper bound: 0.0102361
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0104214, upper bound: 0.0106635
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102694, upper bound: 0.0108068
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0109667, upper bound: 0.0101659
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0107489, upper bound: 0.0102894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102867, upper bound: 0.0106458
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0100897, upper bound: 0.0107654
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0106799, upper bound: 0.0100907
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0105424, upper bound: 0.0102868
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102793, upper bound: 0.0108329
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0101540, upper bound: 0.0110132
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0107117, upper bound: 0.0102804
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0105562, upper bound: 0.0104295
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102317, upper bound: 0.0108316
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0110076
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0109544, upper bound: 0.0100730
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0107257, upper bound: 0.0102442
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0103975, upper bound: 0.0106638
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102228, upper bound: 0.0108068
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0109667, upper bound: 0.0101914
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0107272, upper bound: 0.0103032
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0102760, upper bound: 0.0106477
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 7, lower bound: -0.0100571, upper bound: 0.0107654

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0098925
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106098, upper bound: 0.0098928
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101073
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104951, upper bound: 0.0101062
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101133, upper bound: 0.0105764
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101458, upper bound: 0.0105025
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099990, upper bound: 0.0108062
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100350, upper bound: 0.0107557
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106161, upper bound: 0.0100747
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106470, upper bound: 0.0100177
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0102463
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105069, upper bound: 0.0101908
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100734, upper bound: 0.0105751
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100744, upper bound: 0.0104749
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099000, upper bound: 0.0107909
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099000, upper bound: 0.0106917
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107732, upper bound: 0.0098819
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108426, upper bound: 0.0098821
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100637
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106737, upper bound: 0.0100623
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100593, upper bound: 0.0105548
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101249, upper bound: 0.0105190
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108142, upper bound: 0.0099974
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108530, upper bound: 0.0099732
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106197, upper bound: 0.0101230
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106745, upper bound: 0.0100963
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0105298
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0104322
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0099165
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106098, upper bound: 0.0099165
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101186
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104940, upper bound: 0.0101147
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101057, upper bound: 0.0105918
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101319, upper bound: 0.0105102
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099852, upper bound: 0.0108062
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100086, upper bound: 0.0107557
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106161, upper bound: 0.0101157
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106470, upper bound: 0.0100430
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0102655
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105066, upper bound: 0.0101983
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100662, upper bound: 0.0105910
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100664, upper bound: 0.0104838
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098840, upper bound: 0.0107909
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098832, upper bound: 0.0106917
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107732, upper bound: 0.0098981
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108426, upper bound: 0.0098980
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100721
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106736, upper bound: 0.0100701
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100325, upper bound: 0.0105558
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100856, upper bound: 0.0105190
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108142, upper bound: 0.0100251
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108530, upper bound: 0.0099869
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106197, upper bound: 0.0101369
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106742, upper bound: 0.0101035
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098941, upper bound: 0.0105327
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0104325
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104325, upper bound: 0.0098934
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105327, upper bound: 0.0098941
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101035, upper bound: 0.0106742
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101369, upper bound: 0.0106197
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099869, upper bound: 0.0108530
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100251, upper bound: 0.0108142
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105190, upper bound: 0.0100856
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105558, upper bound: 0.0100325
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0106736
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100721, upper bound: 0.0106032
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098980, upper bound: 0.0108426
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098981, upper bound: 0.0107732
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0098832
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107909, upper bound: 0.0098840
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104838, upper bound: 0.0100664
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105910, upper bound: 0.0100662
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101983, upper bound: 0.0105066
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102655, upper bound: 0.0104737
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100430, upper bound: 0.0106470
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101157, upper bound: 0.0106161
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107557, upper bound: 0.0100086
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108062, upper bound: 0.0099852
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105102, upper bound: 0.0101319
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105918, upper bound: 0.0101057
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101147, upper bound: 0.0104940
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101186, upper bound: 0.0104316
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099165, upper bound: 0.0106098
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099165, upper bound: 0.0105259
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104322, upper bound: 0.0099168
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105298, upper bound: 0.0099168
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100963, upper bound: 0.0106745
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101230, upper bound: 0.0106197
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099732, upper bound: 0.0108530
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099974, upper bound: 0.0108142
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105190, upper bound: 0.0101249
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105548, upper bound: 0.0100593
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100623, upper bound: 0.0106737
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100637, upper bound: 0.0106032
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098821, upper bound: 0.0108426
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0107732
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0099000
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107909, upper bound: 0.0099000
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104749, upper bound: 0.0100744
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105751, upper bound: 0.0100734
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101908, upper bound: 0.0105069
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102463, upper bound: 0.0104737
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100177, upper bound: 0.0106470
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100747, upper bound: 0.0106161
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107557, upper bound: 0.0100350
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108062, upper bound: 0.0099990
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105024, upper bound: 0.0101458
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105764, upper bound: 0.0101133
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101062, upper bound: 0.0104951
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101073, upper bound: 0.0104316
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0106098
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098925, upper bound: 0.0105259
time: 1.28 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0098925
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106098, upper bound: 0.0098928
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101073
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0104951, upper bound: 0.0101062
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101133, upper bound: 0.0105764
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101458, upper bound: 0.0105025
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099990, upper bound: 0.0108062
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100350, upper bound: 0.0107557
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106161, upper bound: 0.0100747
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106470, upper bound: 0.0100177
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0102463
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105069, upper bound: 0.0101908
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100734, upper bound: 0.0105751
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100744, upper bound: 0.0104749
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099000, upper bound: 0.0107909
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099000, upper bound: 0.0106917
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0107732, upper bound: 0.0098819
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0108426, upper bound: 0.0098821
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100637
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106737, upper bound: 0.0100623
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100593, upper bound: 0.0105548
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101249, upper bound: 0.0105190
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0108142, upper bound: 0.0099974
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0108530, upper bound: 0.0099732
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106197, upper bound: 0.0101230
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106745, upper bound: 0.0100963
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0105298
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099168, upper bound: 0.0104322
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105259, upper bound: 0.0099165
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106098, upper bound: 0.0099165
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101186
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0104940, upper bound: 0.0101147
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101057, upper bound: 0.0105918
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101319, upper bound: 0.0105102
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099852, upper bound: 0.0108062
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100086, upper bound: 0.0107557
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106161, upper bound: 0.0101157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106470, upper bound: 0.0100430
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0104737, upper bound: 0.0102655
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105066, upper bound: 0.0101983
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100662, upper bound: 0.0105910
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100664, upper bound: 0.0104838
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0098840, upper bound: 0.0107909
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0098832, upper bound: 0.0106917
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0107732, upper bound: 0.0098981
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0108426, upper bound: 0.0098980
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106032, upper bound: 0.0100721
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106736, upper bound: 0.0100701
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100325, upper bound: 0.0105558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100856, upper bound: 0.0105190
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0108142, upper bound: 0.0100251
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0108530, upper bound: 0.0099869
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106197, upper bound: 0.0101369
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106742, upper bound: 0.0101035
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0098941, upper bound: 0.0105327
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0104325
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0104325, upper bound: 0.0098934
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105327, upper bound: 0.0098941
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101035, upper bound: 0.0106742
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101369, upper bound: 0.0106197
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099869, upper bound: 0.0108530
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100251, upper bound: 0.0108142
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105190, upper bound: 0.0100856
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105558, upper bound: 0.0100325
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100701, upper bound: 0.0106736
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100721, upper bound: 0.0106032
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0098980, upper bound: 0.0108426
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0098981, upper bound: 0.0107732
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0098832
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0107909, upper bound: 0.0098840
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0104838, upper bound: 0.0100664
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105910, upper bound: 0.0100662
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101983, upper bound: 0.0105066
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0102655, upper bound: 0.0104737
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100430, upper bound: 0.0106470
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101157, upper bound: 0.0106161
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0107557, upper bound: 0.0100086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0108062, upper bound: 0.0099852
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105102, upper bound: 0.0101319
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105918, upper bound: 0.0101057
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101147, upper bound: 0.0104940
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101186, upper bound: 0.0104316
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099165, upper bound: 0.0106098
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099165, upper bound: 0.0105259
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0104322, upper bound: 0.0099168
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105298, upper bound: 0.0099168
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100963, upper bound: 0.0106745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101230, upper bound: 0.0106197
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099732, upper bound: 0.0108530
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0099974, upper bound: 0.0108142
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105190, upper bound: 0.0101249
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105548, upper bound: 0.0100593
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100623, upper bound: 0.0106737
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100637, upper bound: 0.0106032
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0098821, upper bound: 0.0108426
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0107732
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0099000
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0107909, upper bound: 0.0099000
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0104749, upper bound: 0.0100744
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105751, upper bound: 0.0100734
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101908, upper bound: 0.0105069
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0102463, upper bound: 0.0104737
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100177, upper bound: 0.0106470
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0100747, upper bound: 0.0106161
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0107557, upper bound: 0.0100350
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0108062, upper bound: 0.0099990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105024, upper bound: 0.0101458
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0105764, upper bound: 0.0101133
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101062, upper bound: 0.0104951
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0101073, upper bound: 0.0104316
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0098819, upper bound: 0.0106098
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 7, lower bound: -0.0098925, upper bound: 0.0105259

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111429, 0.0111135
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111187, 0.0111355
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111429, 0.0111135
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111187, 0.0111355
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111429, 0.0111135
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0077788, 0.0051970, -0.0077788, 0.0051970, -0.0129758, 0.0129758
1: -0.0061906, -0.0010504, -0.0061906, -0.0010504, -0.0051403, 0.0051403
2: 0.0274329, 0.0405613, 0.0274329, 0.0405613, -0.0131284, 0.0131284
3: -0.0074004, 0.0057227, -0.0074004, 0.0057227, -0.0111187, 0.0111355
4: -0.0057486, 0.0056034, -0.0057486, 0.0056034, -0.0113521, 0.0113521
5: 0.0066687, 0.0167235, 0.0066687, 0.0167235, -0.0100548, 0.0100548
6: -0.0118994, 0.0026317, -0.0118994, 0.0026317, -0.0145312, 0.0145312
7: 0.9658857, 0.9843481, 0.9658857, 0.9843481, -0.0184624, 0.0184624
8: -0.0224414, -0.0004651, -0.0224414, -0.0004651, -0.0219763, 0.0219763
9: -0.0044427, 0.0089871, -0.0044427, 0.0089871, -0.0134298, 0.0134298

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
time: 1.02 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094755
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097711, upper bound: 0.0094706
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0093490
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097746, upper bound: 0.0095060
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094482
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098403, upper bound: 0.0094448
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0097750, upper bound: 0.0095454
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095460, upper bound: 0.0097750
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094482, upper bound: 0.0098403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0095060, upper bound: 0.0097746
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0093490, upper bound: 0.0098403
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094615
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 7, lower bound: -0.0098358, upper bound: 0.0094592

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.13 + 591.65 = 594.78 seconds
