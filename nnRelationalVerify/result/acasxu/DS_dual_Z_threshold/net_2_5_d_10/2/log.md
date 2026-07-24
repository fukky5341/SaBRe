## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.857701161


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124)
1: (-0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204)
2: (-0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664)
3: (-0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837)
4: (-0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.07 + 0.97 = 3.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.8707626, upper bound: 0.8707626

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8701518, upper bound: 0.8701518
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8701518, upper bound: 0.8704819
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.80 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -0.8701518, upper bound: 0.8701518
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -0.8701518, upper bound: 0.8704819

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698820, upper bound: 0.8701518
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8704819, upper bound: 0.8700655
time: 0.30 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8700655, upper bound: 0.8704819
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8701518, upper bound: 0.8698820
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.68 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.8698820, upper bound: 0.8701518
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.8704819, upper bound: 0.8700655
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.8700655, upper bound: 0.8704819
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.8701518, upper bound: 0.8698820

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8640182
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8639606
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637426, upper bound: 0.8637163
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640182, upper bound: 0.8637163
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637163, upper bound: 0.8640182
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637163, upper bound: 0.8637426
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8639606, upper bound: 0.8637155
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640182, upper bound: 0.8637155
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.70 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8640182
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8639606
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.8637426, upper bound: 0.8637163
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.8640182, upper bound: 0.8637163
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.8637163, upper bound: 0.8640182
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.8637163, upper bound: 0.8637426
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.8639606, upper bound: 0.8637155
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.8640182, upper bound: 0.8637155

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8637622
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8640097
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8637479
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8639132
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637370, upper bound: 0.8637155
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637426, upper bound: 0.8637163
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640097, upper bound: 0.8637155
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637622, upper bound: 0.8637163
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637163, upper bound: 0.8637622
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8640097
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637163, upper bound: 0.8637426
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8637370
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8639132, upper bound: 0.8637155
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637479, upper bound: 0.8637155
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640097, upper bound: 0.8637155
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637622, upper bound: 0.8637155
time: 0.33 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.75 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8637622
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8640097
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8637479
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8639132
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637370, upper bound: 0.8637155
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637426, upper bound: 0.8637163
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8640097, upper bound: 0.8637155
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637622, upper bound: 0.8637163
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637163, upper bound: 0.8637622
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8640097
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637163, upper bound: 0.8637426
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637155, upper bound: 0.8637370
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8639132, upper bound: 0.8637155
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637479, upper bound: 0.8637155
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8640097, upper bound: 0.8637155
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.8637622, upper bound: 0.8637155

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8575361
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8574405
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8575361, upper bound: 0.8573228
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573722
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573722, upper bound: 0.8573228
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8575361
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8574405, upper bound: 0.8573228
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8575361, upper bound: 0.8573228
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8575361
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8574405
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8575361, upper bound: 0.8573228
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573722
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573722, upper bound: 0.8573228
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8575361
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8574405, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8575361, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.04 + 89.61 = 92.65 seconds
