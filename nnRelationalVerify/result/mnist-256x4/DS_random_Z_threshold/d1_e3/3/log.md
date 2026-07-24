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
execution time: IAR + RelationalAnalysis = 0.98 + 3.10 = 4.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0882511, upper bound: 0.0882511

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0808674, upper bound: 0.0808674
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0808674, upper bound: 0.0808674
time: 1.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.17 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.17
Output dim: 9, lower bound: -0.0808674, upper bound: 0.0808674
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.17
Output dim: 9, lower bound: -0.0808674, upper bound: 0.0808674

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0765231, upper bound: 0.0787599
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787599, upper bound: 0.0765231
time: 1.09 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0804513, upper bound: 0.0784977
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0784977, upper bound: 0.0804513
time: 1.35 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.73 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 9, lower bound: -0.0765231, upper bound: 0.0787599
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 9, lower bound: -0.0787599, upper bound: 0.0765231
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 9, lower bound: -0.0804513, upper bound: 0.0784977
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 9, lower bound: -0.0784977, upper bound: 0.0804513

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757178, upper bound: 0.0769544
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0745216, upper bound: 0.0778706
time: 1.05 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783531, upper bound: 0.0742733
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0764271, upper bound: 0.0761180
time: 0.99 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0801957, upper bound: 0.0784629
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0804180, upper bound: 0.0784009
time: 1.40 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0777270, upper bound: 0.0796069
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0777182, upper bound: 0.0796803
time: 2.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.64 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 9, lower bound: -0.0757178, upper bound: 0.0769544
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 9, lower bound: -0.0745216, upper bound: 0.0778706
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 9, lower bound: -0.0783531, upper bound: 0.0742733
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 9, lower bound: -0.0764271, upper bound: 0.0761180
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 9, lower bound: -0.0801957, upper bound: 0.0784629
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 9, lower bound: -0.0804180, upper bound: 0.0784009
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 9, lower bound: -0.0777270, upper bound: 0.0796069
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 9, lower bound: -0.0777182, upper bound: 0.0796803

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416468, 0.0417546
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0753139, upper bound: 0.0746504
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0734418, upper bound: 0.0765481
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0416146
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740147, upper bound: 0.0770847
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0738462, upper bound: 0.0773837
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0774628, upper bound: 0.0724392
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0765481, upper bound: 0.0734418
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0758914, upper bound: 0.0754435
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0758190, upper bound: 0.0756180
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0759124, upper bound: 0.0756842
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773776, upper bound: 0.0740586
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0796471, upper bound: 0.0776133
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0795732, upper bound: 0.0776254
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0735050, upper bound: 0.0774755
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0756198, upper bound: 0.0753409
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0735050, upper bound: 0.0775434
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0756055, upper bound: 0.0753475
time: 1.11 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0753139, upper bound: 0.0746504
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0734418, upper bound: 0.0765481
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0740147, upper bound: 0.0770847
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0738462, upper bound: 0.0773837
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0774628, upper bound: 0.0724392
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0765481, upper bound: 0.0734418
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0758914, upper bound: 0.0754435
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0758190, upper bound: 0.0756180
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0759124, upper bound: 0.0756842
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0773776, upper bound: 0.0740586
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0796471, upper bound: 0.0776133
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0795732, upper bound: 0.0776254
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0735050, upper bound: 0.0774755
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0756198, upper bound: 0.0753409
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0735050, upper bound: 0.0775434
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 9, lower bound: -0.0756055, upper bound: 0.0753475

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0415331, 0.0417243
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0707160, upper bound: 0.0718175
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0726662, upper bound: 0.0704449
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416062, 0.0416530
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0726979, upper bound: 0.0754987
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725536, upper bound: 0.0757983
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416083, 0.0414801
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0732768, upper bound: 0.0761628
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0729917, upper bound: 0.0763268
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416538, 0.0414430
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0694591, upper bound: 0.0746152
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0711330, upper bound: 0.0729934
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0414925, 0.0417447
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0730770, upper bound: 0.0698240
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0746967, upper bound: 0.0680402
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416427, 0.0416100
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0763434, upper bound: 0.0734082
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0765141, upper bound: 0.0731236
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751425, upper bound: 0.0745380
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749598, upper bound: 0.0746892
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0750084, upper bound: 0.0748418
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749977, upper bound: 0.0748498
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751730, upper bound: 0.0747751
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0750027, upper bound: 0.0749354
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0765106, upper bound: 0.0724608
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0754211, upper bound: 0.0731299
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0791629, upper bound: 0.0769642
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788554, upper bound: 0.0770919
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788221, upper bound: 0.0767164
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786415, upper bound: 0.0768907
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0731704, upper bound: 0.0774421
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0734717, upper bound: 0.0772658
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0747338, upper bound: 0.0733335
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0738616, upper bound: 0.0745352
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0731704, upper bound: 0.0775101
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0734717, upper bound: 0.0773208
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0712051, upper bound: 0.0726505
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0728029, upper bound: 0.0707805
time: 1.53 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0707160, upper bound: 0.0718175
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0726662, upper bound: 0.0704449
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0726979, upper bound: 0.0754987
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0725536, upper bound: 0.0757983
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0732768, upper bound: 0.0761628
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0729917, upper bound: 0.0763268
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0694591, upper bound: 0.0746152
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0711330, upper bound: 0.0729934
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0730770, upper bound: 0.0698240
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0746967, upper bound: 0.0680402
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0763434, upper bound: 0.0734082
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0765141, upper bound: 0.0731236
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0751425, upper bound: 0.0745380
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0749598, upper bound: 0.0746892
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0750084, upper bound: 0.0748418
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0749977, upper bound: 0.0748498
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0751730, upper bound: 0.0747751
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0750027, upper bound: 0.0749354
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0765106, upper bound: 0.0724608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0754211, upper bound: 0.0731299
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0791629, upper bound: 0.0769642
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0788554, upper bound: 0.0770919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0788221, upper bound: 0.0767164
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0786415, upper bound: 0.0768907
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0731704, upper bound: 0.0774421
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0734717, upper bound: 0.0772658
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0747338, upper bound: 0.0733335
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0738616, upper bound: 0.0745352
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0731704, upper bound: 0.0775101
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0734717, upper bound: 0.0773208
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0712051, upper bound: 0.0726505
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 9, lower bound: -0.0728029, upper bound: 0.0707805

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0413036, 0.0413320
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0681884, upper bound: 0.0726434
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0701157, upper bound: 0.0713620
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412852, 0.0413668
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0680914, upper bound: 0.0729100
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0699393, upper bound: 0.0715779
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0413165, 0.0411555
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0728670, upper bound: 0.0740297
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0711747, upper bound: 0.0757527
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412837, 0.0411831
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0722215, upper bound: 0.0754586
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0722215, upper bound: 0.0755170
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416285, 0.0415990
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0755402, upper bound: 0.0726283
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0754976, upper bound: 0.0726283
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416305, 0.0415958
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757091, upper bound: 0.0723429
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0756682, upper bound: 0.0723429
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416638, 0.0416544
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0742485, upper bound: 0.0724608
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0733706, upper bound: 0.0737608
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0743760, upper bound: 0.0739436
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0743496, upper bound: 0.0739436
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417355, 0.0417546
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0756614, upper bound: 0.0716623
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0756382, upper bound: 0.0716909
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749568, upper bound: 0.0725250
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0745210, upper bound: 0.0726001
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0784113, upper bound: 0.0760496
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0781995, upper bound: 0.0762265
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0779703, upper bound: 0.0753196
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0770340, upper bound: 0.0762176
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0744958, upper bound: 0.0738241
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0759713, upper bound: 0.0724351
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0743284, upper bound: 0.0740392
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757755, upper bound: 0.0725259
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0726581, upper bound: 0.0766672
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725123, upper bound: 0.0769545
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0727401, upper bound: 0.0763228
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725669, upper bound: 0.0765179
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0724320, upper bound: 0.0765463
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0722601, upper bound: 0.0767625
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0726283, upper bound: 0.0755402
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716073, upper bound: 0.0764159
time: 1.05 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0681884, upper bound: 0.0726434
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0701157, upper bound: 0.0713620
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0680914, upper bound: 0.0729100
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0699393, upper bound: 0.0715779
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0728670, upper bound: 0.0740297
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0711747, upper bound: 0.0757527
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0722215, upper bound: 0.0754586
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0722215, upper bound: 0.0755170
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0755402, upper bound: 0.0726283
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0754976, upper bound: 0.0726283
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0757091, upper bound: 0.0723429
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0756682, upper bound: 0.0723429
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0742485, upper bound: 0.0724608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0733706, upper bound: 0.0737608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0743760, upper bound: 0.0739436
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0743496, upper bound: 0.0739436
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0756614, upper bound: 0.0716623
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0756382, upper bound: 0.0716909
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0749568, upper bound: 0.0725250
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0745210, upper bound: 0.0726001
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0784113, upper bound: 0.0760496
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0781995, upper bound: 0.0762265
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0779703, upper bound: 0.0753196
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0770340, upper bound: 0.0762176
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0744958, upper bound: 0.0738241
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0759713, upper bound: 0.0724351
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0743284, upper bound: 0.0740392
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0757755, upper bound: 0.0725259
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0726581, upper bound: 0.0766672
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0725123, upper bound: 0.0769545
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0727401, upper bound: 0.0763228
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0725669, upper bound: 0.0765179
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0724320, upper bound: 0.0765463
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0722601, upper bound: 0.0767625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0726283, upper bound: 0.0755402
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.91
Output dim: 9, lower bound: -0.0716073, upper bound: 0.0764159

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412700, 0.0410420
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708878, upper bound: 0.0757191
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0711416, upper bound: 0.0753832
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412750, 0.0411746
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718125, upper bound: 0.0754255
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0721888, upper bound: 0.0750860
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412753, 0.0411757
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0718099, upper bound: 0.0733807
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701625, upper bound: 0.0751075
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416206, 0.0415904
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0750751, upper bound: 0.0720316
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0746003, upper bound: 0.0720982
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416200, 0.0415903
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0750365, upper bound: 0.0720316
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0745483, upper bound: 0.0720982
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416226, 0.0415872
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749648, upper bound: 0.0714548
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0746813, upper bound: 0.0716003
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416220, 0.0415873
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0714476, upper bound: 0.0696728
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0727309, upper bound: 0.0679192
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417277, 0.0417546
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0748997, upper bound: 0.0707955
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0747134, upper bound: 0.0709232
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417268, 0.0417546
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0713471, upper bound: 0.0696064
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0735427, upper bound: 0.0671966
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0775099, upper bound: 0.0742502
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0765971, upper bound: 0.0751857
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0738468, upper bound: 0.0733836
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0753129, upper bound: 0.0718930
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0738223, upper bound: 0.0732316
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0758326, upper bound: 0.0708031
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0726197, upper bound: 0.0741071
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749141, upper bound: 0.0718361
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0750812, upper bound: 0.0707584
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0740848, upper bound: 0.0715095
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0748829, upper bound: 0.0709113
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0738376, upper bound: 0.0715894
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0682456, upper bound: 0.0738311
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0699830, upper bound: 0.0723596
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716866, upper bound: 0.0751869
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705949, upper bound: 0.0760608
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417396
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0718902, upper bound: 0.0744582
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708854, upper bound: 0.0754308
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0681140, upper bound: 0.0736624
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0698924, upper bound: 0.0722494
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417414
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0680369, upper bound: 0.0736788
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0697484, upper bound: 0.0722298
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0679182, upper bound: 0.0738996
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0695330, upper bound: 0.0724170
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0415904, 0.0416462
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0718902, upper bound: 0.0744865
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0717427, upper bound: 0.0748015
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417252, 0.0414941
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0710667, upper bound: 0.0754657
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0710254, upper bound: 0.0759565
time: 1.01 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.91 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0708878, upper bound: 0.0757191
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0711416, upper bound: 0.0753832
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0718125, upper bound: 0.0754255
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0721888, upper bound: 0.0750860
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0718099, upper bound: 0.0733807
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0701625, upper bound: 0.0751075
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0750751, upper bound: 0.0720316
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0746003, upper bound: 0.0720982
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0750365, upper bound: 0.0720316
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0745483, upper bound: 0.0720982
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0749648, upper bound: 0.0714548
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0746813, upper bound: 0.0716003
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0714476, upper bound: 0.0696728
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0727309, upper bound: 0.0679192
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0748997, upper bound: 0.0707955
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0747134, upper bound: 0.0709232
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0713471, upper bound: 0.0696064
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0735427, upper bound: 0.0671966
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0775099, upper bound: 0.0742502
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0765971, upper bound: 0.0751857
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0738468, upper bound: 0.0733836
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0753129, upper bound: 0.0718930
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0738223, upper bound: 0.0732316
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0758326, upper bound: 0.0708031
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0726197, upper bound: 0.0741071
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0749141, upper bound: 0.0718361
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0750812, upper bound: 0.0707584
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0740848, upper bound: 0.0715095
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0748829, upper bound: 0.0709113
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0738376, upper bound: 0.0715894
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0682456, upper bound: 0.0738311
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0699830, upper bound: 0.0723596
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0716866, upper bound: 0.0751869
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0705949, upper bound: 0.0760608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0718902, upper bound: 0.0744582
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0708854, upper bound: 0.0754308
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0681140, upper bound: 0.0736624
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0698924, upper bound: 0.0722494
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0680369, upper bound: 0.0736788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0697484, upper bound: 0.0722298
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0679182, upper bound: 0.0738996
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0695330, upper bound: 0.0724170
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0718902, upper bound: 0.0744865
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0717427, upper bound: 0.0748015
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0710667, upper bound: 0.0754657
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 9, lower bound: -0.0710254, upper bound: 0.0759565

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412596, 0.0410343
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665305, upper bound: 0.0729666
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0681875, upper bound: 0.0713643
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412611, 0.0410316
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0703463, upper bound: 0.0745533
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0703463, upper bound: 0.0745853
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412627, 0.0411639
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0714000, upper bound: 0.0733272
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0697950, upper bound: 0.0750099
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412664, 0.0411624
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0717774, upper bound: 0.0732329
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0701291, upper bound: 0.0746675
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412363, 0.0410621
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 190

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0697962, upper bound: 0.0750743
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0701292, upper bound: 0.0747158
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0414544, 0.0414765
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0743354, upper bound: 0.0711595
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0740099, upper bound: 0.0712976
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0414537, 0.0414761
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 190

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0708482, upper bound: 0.0694087
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0720227, upper bound: 0.0675201
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0414355, 0.0416859
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0730984, upper bound: 0.0712650
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0746356, upper bound: 0.0701241
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0415905, 0.0415400
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0720580, upper bound: 0.0730656
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0744849, upper bound: 0.0708078
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0711396, upper bound: 0.0697905
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0731853, upper bound: 0.0674136
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0413522, 0.0415709
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0750743, upper bound: 0.0697962
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749230, upper bound: 0.0700883
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0414358, 0.0417088
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0746023, upper bound: 0.0701323
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0743060, upper bound: 0.0702239
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0414721, 0.0414817
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0709468, upper bound: 0.0741519
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0708078, upper bound: 0.0744408
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416019, 0.0413282
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0698755, upper bound: 0.0751080
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0696456, upper bound: 0.0753113
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0414252, 0.0411727
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Candidate
type: DSZ, layer: 1, pos: 190

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0703463, upper bound: 0.0745533
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0703033, upper bound: 0.0749621
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0415589, 0.0413649
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0703463, upper bound: 0.0745853
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0701292, upper bound: 0.0747158
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0416074, 0.0413279
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0703033, upper bound: 0.0750033
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700901, upper bound: 0.0752118
time: 1.06 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.00 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0665305, upper bound: 0.0729666
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0681875, upper bound: 0.0713643
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0703463, upper bound: 0.0745533
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0703463, upper bound: 0.0745853
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0714000, upper bound: 0.0733272
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0697950, upper bound: 0.0750099
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0717774, upper bound: 0.0732329
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0701291, upper bound: 0.0746675
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0697962, upper bound: 0.0750743
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0701292, upper bound: 0.0747158
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0743354, upper bound: 0.0711595
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0740099, upper bound: 0.0712976
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0708482, upper bound: 0.0694087
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0720227, upper bound: 0.0675201
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0730984, upper bound: 0.0712650
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0746356, upper bound: 0.0701241
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0720580, upper bound: 0.0730656
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0744849, upper bound: 0.0708078
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0711396, upper bound: 0.0697905
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0731853, upper bound: 0.0674136
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0750743, upper bound: 0.0697962
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0749230, upper bound: 0.0700883
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0746023, upper bound: 0.0701323
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0743060, upper bound: 0.0702239
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0709468, upper bound: 0.0741519
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0708078, upper bound: 0.0744408
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0698755, upper bound: 0.0751080
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0696456, upper bound: 0.0753113
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0703463, upper bound: 0.0745533
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0703033, upper bound: 0.0749621
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0703463, upper bound: 0.0745853
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0701292, upper bound: 0.0747158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0703033, upper bound: 0.0750033
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.00
Output dim: 9, lower bound: -0.0700901, upper bound: 0.0752118

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412258, 0.0410532
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0655487, upper bound: 0.0722316
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0670322, upper bound: 0.0706779
time: 2.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0410532, 0.0412449
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0706779, upper bound: 0.0670322
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0722316, upper bound: 0.0655487
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412953, 0.0410021
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0656129, upper bound: 0.0722728
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0671391, upper bound: 0.0707970
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412760, 0.0410331
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0654473, upper bound: 0.0724835
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0668604, upper bound: 0.0709470
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0412815, 0.0410339
1: -0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090
2: 0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664
3: -0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061
4: -0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074
5: -0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746
6: -0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411
7: -0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521
8: -0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921
9: 0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0657491, upper bound: 0.0723486
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0673908, upper bound: 0.0708837
time: 1.07 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 3.37 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 9, lower bound: -0.0655487, upper bound: 0.0722316
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 9, lower bound: -0.0670322, upper bound: 0.0706779
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 9, lower bound: -0.0706779, upper bound: 0.0670322
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 9, lower bound: -0.0722316, upper bound: 0.0655487
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 9, lower bound: -0.0656129, upper bound: 0.0722728
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 9, lower bound: -0.0671391, upper bound: 0.0707970
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 9, lower bound: -0.0654473, upper bound: 0.0724835
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 9, lower bound: -0.0668604, upper bound: 0.0709470
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 9, lower bound: -0.0657491, upper bound: 0.0723486
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 9, lower bound: -0.0673908, upper bound: 0.0708837

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.08 + 359.01 = 363.10 seconds
