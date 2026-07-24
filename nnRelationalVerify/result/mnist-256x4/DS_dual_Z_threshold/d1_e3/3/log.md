## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.075013435


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417546)
1: (-0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090)
2: (0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664)
3: (-0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061)
4: (-0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074)
5: (-0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746)
6: (-0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411)
7: (-0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521)
8: (-0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921)
9: (0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 3.06 = 4.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0882511, upper bound: 0.0882511

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0834813, upper bound: 0.0834932
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0834932, upper bound: 0.0834813
time: 1.71 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.19 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.19
Output dim: 9, lower bound: -0.0834813, upper bound: 0.0834932
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.19
Output dim: 9, lower bound: -0.0834932, upper bound: 0.0834813

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417546
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0827806, upper bound: 0.0825677
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0825633, upper bound: 0.0827926
time: 1.32 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417546
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0827926, upper bound: 0.0825633
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0825677, upper bound: 0.0827806
time: 1.79 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.13 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 9, lower bound: -0.0827806, upper bound: 0.0825677
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 9, lower bound: -0.0825633, upper bound: 0.0827926
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 9, lower bound: -0.0827926, upper bound: 0.0825633
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 9, lower bound: -0.0825677, upper bound: 0.0827806

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417546
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0692547, upper bound: 0.0689628
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0692547, upper bound: 0.0689628
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417546
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0689628, upper bound: 0.0692547
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0689628, upper bound: 0.0692547
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417546
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0692547, upper bound: 0.0689628
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0692547, upper bound: 0.0689628
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417546
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0689628, upper bound: 0.0692547
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0689628, upper bound: 0.0692547
time: 1.22 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.60 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.60
Output dim: 9, lower bound: -0.0692547, upper bound: 0.0689628
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.60
Output dim: 9, lower bound: -0.0692547, upper bound: 0.0689628
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.60
Output dim: 9, lower bound: -0.0689628, upper bound: 0.0692547
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.60
Output dim: 9, lower bound: -0.0689628, upper bound: 0.0692547
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.60
Output dim: 9, lower bound: -0.0692547, upper bound: 0.0689628
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.60
Output dim: 9, lower bound: -0.0692547, upper bound: 0.0689628
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.60
Output dim: 9, lower bound: -0.0689628, upper bound: 0.0692547
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.60
Output dim: 9, lower bound: -0.0689628, upper bound: 0.0692547

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.46 + 32.94 = 37.41 seconds
