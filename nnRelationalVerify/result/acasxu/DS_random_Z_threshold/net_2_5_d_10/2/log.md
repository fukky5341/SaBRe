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
execution time: IAR + RelationalAnalysis = 0.77 + 0.85 = 1.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.8707626, upper bound: 0.8707626

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8702310, upper bound: 0.8707626
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8702310, upper bound: 0.8702310
time: 0.17 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.38 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.38
Output dim: 0, lower bound: -0.8702310, upper bound: 0.8707626
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.38
Output dim: 0, lower bound: -0.8702310, upper bound: 0.8702310

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8664945, upper bound: 0.8669179
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8665270, upper bound: 0.8666538
time: 0.20 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8687886, upper bound: 0.8687886
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8687886, upper bound: 0.8687886
time: 0.20 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.11 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.11
Output dim: 0, lower bound: -0.8664945, upper bound: 0.8669179
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.11
Output dim: 0, lower bound: -0.8665270, upper bound: 0.8666538
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.11
Output dim: 0, lower bound: -0.8687886, upper bound: 0.8687886
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.11
Output dim: 0, lower bound: -0.8687886, upper bound: 0.8687886

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614201
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614206
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8662597, upper bound: 0.8662597
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8663219, upper bound: 0.8663305
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8663012, upper bound: 0.8663012
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8665581, upper bound: 0.8663012
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8690378, upper bound: 0.8683921
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8687220, upper bound: 0.8682766
time: 0.21 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.13 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.13
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614201
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.13
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614206
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.13
Output dim: 0, lower bound: -0.8662597, upper bound: 0.8662597
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.13
Output dim: 0, lower bound: -0.8663219, upper bound: 0.8663305
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.13
Output dim: 0, lower bound: -0.8663012, upper bound: 0.8663012
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.13
Output dim: 0, lower bound: -0.8665581, upper bound: 0.8663012
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.13
Output dim: 0, lower bound: -0.8690378, upper bound: 0.8683921
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.13
Output dim: 0, lower bound: -0.8687220, upper bound: 0.8682766

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8611954, upper bound: 0.8613045
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614201
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614206
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614206
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573553, upper bound: 0.8573553
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573553, upper bound: 0.8573553
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8574096, upper bound: 0.8573553
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573553, upper bound: 0.8573553
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8646125, upper bound: 0.8646031
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8646125, upper bound: 0.8646031
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647575, upper bound: 0.8645257
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647575, upper bound: 0.8645257
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8659687, upper bound: 0.8659434
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8662099, upper bound: 0.8659429
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8659540, upper bound: 0.8659429
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8660066, upper bound: 0.8659429
time: 0.19 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8611954, upper bound: 0.8613045
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614201
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614206
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8611934, upper bound: 0.8614206
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8573553, upper bound: 0.8573553
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8573553, upper bound: 0.8573553
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8574096, upper bound: 0.8573553
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8573553, upper bound: 0.8573553
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8646125, upper bound: 0.8646031
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8646125, upper bound: 0.8646031
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8647575, upper bound: 0.8645257
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8647575, upper bound: 0.8645257
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8659687, upper bound: 0.8659434
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8662099, upper bound: 0.8659429
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8659540, upper bound: 0.8659429
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.8660066, upper bound: 0.8659429

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601951
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601425
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8602010
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601137
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8602365
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8603743
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599168, upper bound: 0.8599044
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8645407, upper bound: 0.8645257
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8645426, upper bound: 0.8645257
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647329, upper bound: 0.8645257
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647575, upper bound: 0.8645257
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647329, upper bound: 0.8645257
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647575, upper bound: 0.8645257
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573722
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
time: 0.19 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601951
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601425
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8602010
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601137
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8602365
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8603743
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8599168, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8645407, upper bound: 0.8645257
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8645426, upper bound: 0.8645257
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8647329, upper bound: 0.8645257
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8647575, upper bound: 0.8645257
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8647329, upper bound: 0.8645257
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8647575, upper bound: 0.8645257
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573722
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.37
Output dim: 0, lower bound: -0.8573228, upper bound: 0.8573228

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601951
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601897
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8600978
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601425
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8602010
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601137
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8602365
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8603743
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599168, upper bound: 0.8599044
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8603743, upper bound: 0.8599044
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8601425, upper bound: 0.8599044
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8602365, upper bound: 0.8599044
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8601897, upper bound: 0.8599044
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8601137, upper bound: 0.8599044
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8600978, upper bound: 0.8599044
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8602010, upper bound: 0.8599044
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8601951, upper bound: 0.8599044
time: 0.20 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601951
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601897
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8600978
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601425
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8602010
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601137
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8602365
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8603743
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599168, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8603743, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8601425, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8602365, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8601897, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8601137, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8600978, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8602010, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.16
Output dim: 0, lower bound: -0.8601951, upper bound: 0.8599044

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592243, upper bound: 0.8600756
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8595828, upper bound: 0.8592352
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8596928, upper bound: 0.8601897
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601101
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8596928, upper bound: 0.8600978
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599740
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8593612
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8593238, upper bound: 0.8597072
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8596928, upper bound: 0.8599044
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8596928
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8557465, upper bound: 0.8559334
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8596928, upper bound: 0.8601137
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8596928, upper bound: 0.8600046
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8592599
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8594280, upper bound: 0.8594661
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8556919, upper bound: 0.8562302
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8557397
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8557011, upper bound: 0.8562580
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8557439
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8594780, upper bound: 0.8593792
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8556894, upper bound: 0.8559334
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8556894, upper bound: 0.8559334
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8557339, upper bound: 0.8559334
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8594280
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8556894, upper bound: 0.8559334
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8556894, upper bound: 0.8559334
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8557473, upper bound: 0.8559334
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8599381, upper bound: 0.8593940
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8593578, upper bound: 0.8594661
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8601773, upper bound: 0.8599044
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8602365, upper bound: 0.8596928
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592344, upper bound: 0.8595828
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8600734, upper bound: 0.8592243
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8557384, upper bound: 0.8559334
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8561049, upper bound: 0.8556894
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592354, upper bound: 0.8595828
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592354, upper bound: 0.8592243
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8601199, upper bound: 0.8599044
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8601951, upper bound: 0.8596928
time: 0.23 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8592243, upper bound: 0.8600756
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8595828, upper bound: 0.8592352
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8596928, upper bound: 0.8601897
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8601101
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8596928, upper bound: 0.8600978
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8599740
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8593612
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8593238, upper bound: 0.8597072
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8596928, upper bound: 0.8599044
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8599044, upper bound: 0.8596928
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8557465, upper bound: 0.8559334
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8596928, upper bound: 0.8601137
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8596928, upper bound: 0.8600046
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8592599
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8594280, upper bound: 0.8594661
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8556919, upper bound: 0.8562302
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8557397
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8557011, upper bound: 0.8562580
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8557439
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8594780, upper bound: 0.8593792
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8556894, upper bound: 0.8559334
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8556894, upper bound: 0.8559334
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8557339, upper bound: 0.8559334
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8594280
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8556894, upper bound: 0.8559334
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8556894, upper bound: 0.8559334
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8557473, upper bound: 0.8559334
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8559334, upper bound: 0.8556894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8599381, upper bound: 0.8593940
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8593578, upper bound: 0.8594661
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8601773, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8602365, upper bound: 0.8596928
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8592344, upper bound: 0.8595828
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8600734, upper bound: 0.8592243
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8557384, upper bound: 0.8559334
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8561049, upper bound: 0.8556894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8592354, upper bound: 0.8595828
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8592354, upper bound: 0.8592243
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8601199, upper bound: 0.8599044
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.29
Output dim: 0, lower bound: -0.8601951, upper bound: 0.8596928

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8590972
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8596297
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8552169, upper bound: 0.8552169
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8552169, upper bound: 0.8552169
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8593621
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8597538
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592243, upper bound: 0.8598656
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8595828, upper bound: 0.8592344
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594278
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8596628
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592243, upper bound: 0.8597209
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8595828, upper bound: 0.8592292
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8593612
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8592599
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8593175
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8592599
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8593777, upper bound: 0.8592599
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594275
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8596737
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8592599
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8593366, upper bound: 0.8595650
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8549774
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8552169, upper bound: 0.8549774
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8593792
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8592599
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8591038
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8591416, upper bound: 0.8587894
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8592599
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8550213, upper bound: 0.8551202
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8555452, upper bound: 0.8549842
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8593578, upper bound: 0.8592599
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8557397, upper bound: 0.8556894
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8562302, upper bound: 0.8556894
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587965, upper bound: 0.8591028
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8591416
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587972, upper bound: 0.8590224
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8591416
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8596787, upper bound: 0.8593472
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
time: 0.21 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.28 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8590972
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8596297
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8552169, upper bound: 0.8552169
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8552169, upper bound: 0.8552169
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8593621
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8597538
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592243, upper bound: 0.8598656
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8595828, upper bound: 0.8592344
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594278
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8596628
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592243, upper bound: 0.8597209
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8595828, upper bound: 0.8592292
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8593612
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8592599
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8593175
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8592599
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8593777, upper bound: 0.8592599
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594275
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8596737
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8594661, upper bound: 0.8592599
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8593366, upper bound: 0.8595650
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8549774
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8552169, upper bound: 0.8549774
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8593792
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8592599
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8591038
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8591416, upper bound: 0.8587894
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8592599
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8550213, upper bound: 0.8551202
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8555452, upper bound: 0.8549842
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8593578, upper bound: 0.8592599
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8557397, upper bound: 0.8556894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8562302, upper bound: 0.8556894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8587965, upper bound: 0.8591028
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8591416
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8587972, upper bound: 0.8590224
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8591416
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8596787, upper bound: 0.8593472
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.8592599, upper bound: 0.8594661
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.28
Output dim: 0, lower bound: -0.7864630, upper bound: 0.7864630

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829758
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829758
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8596297
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8594323
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8590371
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8554636
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8550218
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8594220
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8591037
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8595085
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8553452, upper bound: 0.8552169
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8556391, upper bound: 0.8552169
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8551031
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8550258
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8589920
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8552169
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8549774
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8591416, upper bound: 0.8587894
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8550273, upper bound: 0.8549774
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8551205, upper bound: 0.8549774
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8595093
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8591416, upper bound: 0.8587894
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8590538
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8552169
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8549774
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8552169
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8549774
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8591416
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8587971, upper bound: 0.8590209
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8594323, upper bound: 0.8587894
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8552169
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8549774
time: 0.23 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829758
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829758
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8596297
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8594323
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8590371
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8554636
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8550218
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8594220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8591037
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8595085
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7852212, upper bound: 0.7852212
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8553452, upper bound: 0.8552169
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8556391, upper bound: 0.8552169
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8551031
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8550258
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8589920
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8552169
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8549774
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8591416, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8550273, upper bound: 0.8549774
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8551205, upper bound: 0.8549774
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8595093
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8591416, upper bound: 0.8587894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8590538
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8552169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8549774
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7842536, upper bound: 0.7842536
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8552169
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8549774
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8591416
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587894, upper bound: 0.8587894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8587971, upper bound: 0.8590209
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8594323, upper bound: 0.8587894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8552169
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.38
Output dim: 0, lower bound: -0.8549774, upper bound: 0.8549774

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8553823
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8547329
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8551501
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8546016
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8548703
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8547264
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829758
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829758
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8551775
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8547236
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8546104, upper bound: 0.8545035
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549196, upper bound: 0.8545035
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8547169
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8546019
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8549196, upper bound: 0.8545035
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829694
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829694
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 35

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124
1: -0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204
2: -0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664
3: -0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837
4: -0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8546014, upper bound: 0.8545035
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8551740, upper bound: 0.8545035
time: 0.23 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 1.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8553823
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8547329
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8551501
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8546016
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8548703
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8547264
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829758
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829758
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8551775
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8547236
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8546104, upper bound: 0.8545035
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8549196, upper bound: 0.8545035
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8547169
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8546019
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8549196, upper bound: 0.8545035
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829694
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829694
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8545035, upper bound: 0.8545035
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.7829693, upper bound: 0.7829693
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8546014, upper bound: 0.8545035
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.37
Output dim: 0, lower bound: -0.8551740, upper bound: 0.8545035

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.62 + 203.21 = 204.83 seconds
