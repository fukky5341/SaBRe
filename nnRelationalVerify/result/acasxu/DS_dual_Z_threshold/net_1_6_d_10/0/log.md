## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 380.96572808527804


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-120.7598648, 310.2168884, -120.7598648, 310.2168884, -430.9767456, 430.9767456)
1: (-306.0167542, 469.4156189, -306.0167542, 469.4156189, -775.4323730, 775.4323730)
2: (-197.4686890, 458.7374573, -197.4686890, 458.7374573, -656.2061768, 656.2061768)
3: (-330.1389465, 542.4745483, -330.1389465, 542.4745483, -872.6135254, 872.6134644)
4: (-287.0395203, 524.3436890, -287.0395203, 524.3436890, -811.3831177, 811.3831177)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.74 + 1.95 = 2.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -380.9771574, upper bound: 380.9771574

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9770743, upper bound: 380.9770743
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9770743, upper bound: 380.9771574
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.42 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.42
Output dim: 0, lower bound: -380.9770743, upper bound: 380.9770743
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.42
Output dim: 0, lower bound: -380.9770743, upper bound: 380.9771574

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -120.7598648, 310.2168884, -120.7598648, 310.2168884, -430.9767456, 430.9767456
1: -306.0167542, 469.4156189, -306.0167542, 469.4156189, -775.4323730, 775.4323730
2: -197.4686890, 458.7374573, -197.4686890, 458.7374573, -656.2061768, 656.2061768
3: -330.1389465, 542.4745483, -330.1389465, 542.4745483, -872.6135254, 872.6134644
4: -287.0395203, 524.3436890, -287.0395203, 524.3436890, -811.3831177, 811.3831177

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9666454, upper bound: 380.9670526
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9665774, upper bound: 380.9670510
time: 0.70 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -120.7598648, 310.2168884, -120.7598648, 310.2168884, -430.9767456, 430.9767456
1: -306.0167542, 469.4156189, -306.0167542, 469.4156189, -775.4323730, 775.4323730
2: -197.4686890, 458.7374573, -197.4686890, 458.7374573, -656.2061768, 656.2061768
3: -330.1389465, 542.4745483, -330.1389465, 542.4745483, -872.6135254, 872.6134644
4: -287.0395203, 524.3436890, -287.0395203, 524.3436890, -811.3831177, 811.3831177

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9670510, upper bound: 380.9665774
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9665774, upper bound: 380.9666454
time: 0.68 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.14 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -380.9666454, upper bound: 380.9670526
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -380.9665774, upper bound: 380.9670510
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -380.9670510, upper bound: 380.9665774
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -380.9665774, upper bound: 380.9666454

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -120.7598648, 310.2168884, -120.7598648, 310.2168884, -430.9767456, 430.9767456
1: -306.0167542, 469.4156189, -306.0167542, 469.4156189, -775.4323730, 775.4323730
2: -197.4686890, 458.7374573, -197.4686890, 458.7374573, -656.2061768, 656.2061768
3: -330.1389465, 542.4745483, -330.1389465, 542.4745483, -872.6135254, 872.6134644
4: -287.0395203, 524.3436890, -287.0395203, 524.3436890, -811.3831177, 811.3831177

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -120.7598648, 310.2168884, -120.7598648, 310.2168884, -430.9767456, 430.9767456
1: -306.0167542, 469.4156189, -306.0167542, 469.4156189, -775.4323730, 775.4323730
2: -197.4686890, 458.7374573, -197.4686890, 458.7374573, -656.2061768, 656.2061768
3: -330.1389465, 542.4745483, -330.1389465, 542.4745483, -872.6135254, 872.6134644
4: -287.0395203, 524.3436890, -287.0395203, 524.3436890, -811.3831177, 811.3831177

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -120.7598648, 310.2168884, -120.7598648, 310.2168884, -430.9767456, 430.9767456
1: -306.0167542, 469.4156189, -306.0167542, 469.4156189, -775.4323730, 775.4323730
2: -197.4686890, 458.7374573, -197.4686890, 458.7374573, -656.2061768, 656.2061768
3: -330.1389465, 542.4745483, -330.1389465, 542.4745483, -872.6135254, 872.6134644
4: -287.0395203, 524.3436890, -287.0395203, 524.3436890, -811.3831177, 811.3831177

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -120.7598648, 310.2168884, -120.7598648, 310.2168884, -430.9767456, 430.9767456
1: -306.0167542, 469.4156189, -306.0167542, 469.4156189, -775.4323730, 775.4323730
2: -197.4686890, 458.7374573, -197.4686890, 458.7374573, -656.2061768, 656.2061768
3: -330.1389465, 542.4745483, -330.1389465, 542.4745483, -872.6135254, 872.6134644
4: -287.0395203, 524.3436890, -287.0395203, 524.3436890, -811.3831177, 811.3831177

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
time: 0.60 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.21 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -380.9633784, upper bound: 380.9633784

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.69 + 14.67 = 17.36 seconds
