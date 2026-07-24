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
execution time: IAR + RelationalAnalysis = 1.05 + 1.99 = 3.04 seconds
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

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0116533, upper bound: 0.0115087
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0115087, upper bound: 0.0116533
time: 1.10 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.37 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.37
Output dim: 7, lower bound: -0.0116533, upper bound: 0.0115087
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.37
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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
time: 0.95 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907
time: 0.93 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.67 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 7, lower bound: -0.0112907, upper bound: 0.0112509
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 7, lower bound: -0.0112509, upper bound: 0.0112907
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.67
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108165
time: 1.02 seconds

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108166
time: 1.04 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108166, upper bound: 0.0110572
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.10 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108165, upper bound: 0.0110572
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
time: 1.11 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.02 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108165
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 7, lower bound: -0.0108937, upper bound: 0.0110343
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 7, lower bound: -0.0110572, upper bound: 0.0108166
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 7, lower bound: -0.0108166, upper bound: 0.0110572
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 7, lower bound: -0.0110343, upper bound: 0.0108937
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 7, lower bound: -0.0108165, upper bound: 0.0110572
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100521, upper bound: 0.0100928
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100521, upper bound: 0.0100928
time: 1.09 seconds

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108722, upper bound: 0.0106604
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108975, upper bound: 0.0106256
time: 1.00 seconds

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

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108759, upper bound: 0.0108241
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107416, upper bound: 0.0110156
time: 1.25 seconds

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

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100283
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100283
time: 1.08 seconds

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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100283, upper bound: 0.0100928
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100283, upper bound: 0.0100928
time: 1.00 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0110156, upper bound: 0.0107416
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108241, upper bound: 0.0108759
time: 1.24 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107805, upper bound: 0.0110550
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108067, upper bound: 0.0110521
time: 1.10 seconds

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108363, upper bound: 0.0107330
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108754, upper bound: 0.0107048
time: 1.05 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0100521, upper bound: 0.0100928
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0100521, upper bound: 0.0100928
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0108722, upper bound: 0.0106604
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0108975, upper bound: 0.0106256
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0108759, upper bound: 0.0108241
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0107416, upper bound: 0.0110156
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100283
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0100928, upper bound: 0.0100283
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0100283, upper bound: 0.0100928
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0100283, upper bound: 0.0100928
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0110156, upper bound: 0.0107416
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0108241, upper bound: 0.0108759
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0107805, upper bound: 0.0110550
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0108067, upper bound: 0.0110521
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0108363, upper bound: 0.0107330
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 7, lower bound: -0.0108754, upper bound: 0.0107048

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108385, upper bound: 0.0101521
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102358, upper bound: 0.0105837
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108754, upper bound: 0.0101257
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0102998, upper bound: 0.0105486
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100342, upper bound: 0.0100050
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100342, upper bound: 0.0100050
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107185, upper bound: 0.0110109
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107324, upper bound: 0.0110037
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100745, upper bound: 0.0099517
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100745, upper bound: 0.0099555
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107582, upper bound: 0.0103006
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102788, upper bound: 0.0108162
time: 1.15 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106974, upper bound: 0.0103047
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102968, upper bound: 0.0110318
time: 1.32 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100240, upper bound: 0.0100928
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0100240, upper bound: 0.0100928
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108175, upper bound: 0.0105857
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0104535, upper bound: 0.0107152
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108581, upper bound: 0.0106965
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108705, upper bound: 0.0106147
time: 1.11 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.98 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0108385, upper bound: 0.0101521
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0102358, upper bound: 0.0105837
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0108754, upper bound: 0.0101257
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0102998, upper bound: 0.0105486
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0100342, upper bound: 0.0100050
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0100342, upper bound: 0.0100050
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0107185, upper bound: 0.0110109
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0107324, upper bound: 0.0110037
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0100745, upper bound: 0.0099517
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0100745, upper bound: 0.0099555
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0107582, upper bound: 0.0103006
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0102788, upper bound: 0.0108162
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0106974, upper bound: 0.0103047
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0102968, upper bound: 0.0110318
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0100240, upper bound: 0.0100928
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0100240, upper bound: 0.0100928
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0108175, upper bound: 0.0105857
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0104535, upper bound: 0.0107152
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0108581, upper bound: 0.0106965
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.98
Output dim: 7, lower bound: -0.0108705, upper bound: 0.0106147

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095712
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095712
time: 0.99 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095708
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095708
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0105042, upper bound: 0.0108519
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0105661, upper bound: 0.0108105
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0105431, upper bound: 0.0108395
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0105762, upper bound: 0.0107592
time: 1.05 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105210, upper bound: 0.0101432
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0106013, upper bound: 0.0101171
time: 1.01 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0102694, upper bound: 0.0108068
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100897, upper bound: 0.0107654
time: 1.45 seconds

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

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106799, upper bound: 0.0100907
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105424, upper bound: 0.0102868
time: 1.16 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101138, upper bound: 0.0108715
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101406, upper bound: 0.0108328
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0107592, upper bound: 0.0105762
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108105, upper bound: 0.0105663
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097659, upper bound: 0.0097938
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097659, upper bound: 0.0097938
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0108395, upper bound: 0.0105431
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106495, upper bound: 0.0106786
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098577, upper bound: 0.0097960
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098577, upper bound: 0.0097960
time: 0.92 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.72 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095712
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095712
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095708
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0098585, upper bound: 0.0095708
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0105042, upper bound: 0.0108519
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0105661, upper bound: 0.0108105
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0105431, upper bound: 0.0108395
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0105762, upper bound: 0.0107592
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0105210, upper bound: 0.0101432
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0106013, upper bound: 0.0101171
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0102694, upper bound: 0.0108068
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0100897, upper bound: 0.0107654
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0106799, upper bound: 0.0100907
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0105424, upper bound: 0.0102868
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0101138, upper bound: 0.0108715
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0101406, upper bound: 0.0108328
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0107592, upper bound: 0.0105762
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0108105, upper bound: 0.0105663
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0097659, upper bound: 0.0097938
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0097659, upper bound: 0.0097938
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0108395, upper bound: 0.0105431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0106495, upper bound: 0.0106786
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0098577, upper bound: 0.0097960
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.72
Output dim: 7, lower bound: -0.0098577, upper bound: 0.0097960

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101186
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099852, upper bound: 0.0108062
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0104940, upper bound: 0.0101147
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100086, upper bound: 0.0107557
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097195, upper bound: 0.0098387
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0097185, upper bound: 0.0098387
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105066, upper bound: 0.0101983
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0098832, upper bound: 0.0106917
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100430, upper bound: 0.0106470
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0101157, upper bound: 0.0106161
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095540, upper bound: 0.0099957
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095540, upper bound: 0.0099958
time: 0.96 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099691, upper bound: 0.0095565
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0099691, upper bound: 0.0095565
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100963, upper bound: 0.0106745
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0099732, upper bound: 0.0108530
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095634, upper bound: 0.0098585
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095634, upper bound: 0.0098585
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0099000
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0101908, upper bound: 0.0105069
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098394, upper bound: 0.0097125
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098394, upper bound: 0.0097131
time: 0.90 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098387, upper bound: 0.0097185
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098387, upper bound: 0.0097195
time: 0.89 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0105751, upper bound: 0.0100734
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0100747, upper bound: 0.0106161
time: 1.33 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.35 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104316, upper bound: 0.0101186
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0099852, upper bound: 0.0108062
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0104940, upper bound: 0.0101147
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100086, upper bound: 0.0107557
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0097195, upper bound: 0.0098387
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0097185, upper bound: 0.0098387
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105066, upper bound: 0.0101983
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098832, upper bound: 0.0106917
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100430, upper bound: 0.0106470
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101157, upper bound: 0.0106161
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0095540, upper bound: 0.0099957
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0095540, upper bound: 0.0099958
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0099691, upper bound: 0.0095565
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0099691, upper bound: 0.0095565
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100963, upper bound: 0.0106745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0099732, upper bound: 0.0108530
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0095634, upper bound: 0.0098585
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0095634, upper bound: 0.0098585
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0106917, upper bound: 0.0099000
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0101908, upper bound: 0.0105069
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098394, upper bound: 0.0097125
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098394, upper bound: 0.0097131
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098387, upper bound: 0.0097185
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0098387, upper bound: 0.0097195
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0105751, upper bound: 0.0100734
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0100747, upper bound: 0.0106161

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 106
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 183

### Candidate
type: DSZ, layer: 3, pos: 106

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
time: 1.15 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.16 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094592, upper bound: 0.0098358
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094615, upper bound: 0.0098358
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0093503, upper bound: 0.0098333
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094706, upper bound: 0.0097711
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0095454, upper bound: 0.0097750
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094448, upper bound: 0.0098403
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0098333, upper bound: 0.0093503
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.16
Output dim: 7, lower bound: -0.0094755, upper bound: 0.0097711

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.04 + 179.21 = 182.24 seconds
