## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 63.39367995097201


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308)
1: (-271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056)
2: (-152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893)
3: (-189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412)
4: (-109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.06 + 2.05 = 3.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -63.3974838, upper bound: 63.3974838

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3974838, upper bound: 63.3974838
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3974838, upper bound: 63.3974837
time: 0.83 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.67 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -63.3974838, upper bound: 63.3974838
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -63.3974838, upper bound: 63.3974837

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958632
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958632
time: 0.79 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.56 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958632
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958632
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3952958, upper bound: 63.3954425
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3952958, upper bound: 63.3954436
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3882717, upper bound: 63.3883041
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3882717, upper bound: 63.3883041
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3951897, upper bound: 63.3951898
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3951898, upper bound: 63.3951897
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140
time: 0.78 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.42 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -63.3952958, upper bound: 63.3954425
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -63.3952958, upper bound: 63.3954436
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -63.3882717, upper bound: 63.3883041
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -63.3882717, upper bound: 63.3883041
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -63.3951897, upper bound: 63.3951898
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -63.3951898, upper bound: 63.3951897
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3952811, upper bound: 63.3952972
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3952811, upper bound: 63.3954011
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3952922
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3951444
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947738
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947703
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949371
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3952492, upper bound: 63.3952492
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3952492, upper bound: 63.3952492
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3952811, upper bound: 63.3952972
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3952811, upper bound: 63.3954011
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3952922
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3951444
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947738
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947703
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949371
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3958140, upper bound: 63.3958140
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3952492, upper bound: 63.3952492
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -63.3952492, upper bound: 63.3952492

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945300, upper bound: 63.3945327
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945300, upper bound: 63.3945508
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950693, upper bound: 63.3952457
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950693, upper bound: 63.3951172
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3880503, upper bound: 63.3880503
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3880503, upper bound: 63.3880503
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3950751
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3951444
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947703
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947737
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949371
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956042, upper bound: 63.3956042
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956042, upper bound: 63.3956042
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949002, upper bound: 63.3949002
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949002, upper bound: 63.3949002
time: 0.75 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3945300, upper bound: 63.3945327
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3945300, upper bound: 63.3945508
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3950693, upper bound: 63.3952457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3950693, upper bound: 63.3951172
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3880503, upper bound: 63.3880503
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3880503, upper bound: 63.3880503
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3950751
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3951444
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947703
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947737
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949371
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3956042, upper bound: 63.3956042
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3956042, upper bound: 63.3956042
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3949002, upper bound: 63.3949002
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.49
Output dim: 0, lower bound: -63.3949002, upper bound: 63.3949002

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943929, upper bound: 63.3943929
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943929, upper bound: 63.3943929
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3599658, upper bound: 63.3599658
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3599658, upper bound: 63.3599658
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950693, upper bound: 63.3950693
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950693, upper bound: 63.3952457
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3867488, upper bound: 63.3867488
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3867488, upper bound: 63.3867488
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949526, upper bound: 63.3949526
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949526, upper bound: 63.3949526
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3951444
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3950751
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947703
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947703
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947737
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947703
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3945203
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3945203
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949371
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947376, upper bound: 63.3947376
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947376, upper bound: 63.3947376
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3924257, upper bound: 63.3924257
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3924257, upper bound: 63.3924257
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3953461, upper bound: 63.3953461
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3953461, upper bound: 63.3953461
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947665, upper bound: 63.3947665
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947665, upper bound: 63.3947665
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3935925, upper bound: 63.3935925
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3935925, upper bound: 63.3935925
time: 0.77 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3943929, upper bound: 63.3943929
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3943929, upper bound: 63.3943929
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3599658, upper bound: 63.3599658
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3599658, upper bound: 63.3599658
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3950693, upper bound: 63.3950693
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3950693, upper bound: 63.3952457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3867488, upper bound: 63.3867488
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3867488, upper bound: 63.3867488
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3949526, upper bound: 63.3949526
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3949526, upper bound: 63.3949526
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3951444
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3950751
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947703
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947703
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947737
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3947703, upper bound: 63.3947703
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3945203
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3945203, upper bound: 63.3945203
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949371
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3949367, upper bound: 63.3949367
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3947376, upper bound: 63.3947376
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3947376, upper bound: 63.3947376
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3924257, upper bound: 63.3924257
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3924257, upper bound: 63.3924257
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3953461, upper bound: 63.3953461
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3953461, upper bound: 63.3953461
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3947665, upper bound: 63.3947665
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3947665, upper bound: 63.3947665
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3935925, upper bound: 63.3935925
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -63.3935925, upper bound: 63.3935925

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941265, upper bound: 63.3941265
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941265, upper bound: 63.3941265
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3932153, upper bound: 63.3932153
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3932153, upper bound: 63.3932153
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950662, upper bound: 63.3950662
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950662, upper bound: 63.3950662
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949444, upper bound: 63.3949479
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949444, upper bound: 63.3950110
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945424, upper bound: 63.3945424
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945424, upper bound: 63.3945424
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949526, upper bound: 63.3949526
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949526, upper bound: 63.3949526
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940519, upper bound: 63.3940519
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940519, upper bound: 63.3940519
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3950751
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3950751
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945944, upper bound: 63.3945944
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945944, upper bound: 63.3945944
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947662
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943689, upper bound: 63.3943688
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943689, upper bound: 63.3943689
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942043, upper bound: 63.3942043
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942043, upper bound: 63.3942043
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945267, upper bound: 63.3945267
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945267, upper bound: 63.3945267
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945183, upper bound: 63.3945180
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945180, upper bound: 63.3945180
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948706, upper bound: 63.3948706
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948706, upper bound: 63.3948706
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949036
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3931200, upper bound: 63.3931200
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3931200, upper bound: 63.3931200
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948076, upper bound: 63.3948076
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948076, upper bound: 63.3948076
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949004, upper bound: 63.3949004
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949004, upper bound: 63.3949004
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945626, upper bound: 63.3945626
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945626, upper bound: 63.3945626
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946721, upper bound: 63.3946721
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946721, upper bound: 63.3946721
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947322, upper bound: 63.3947322
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947322, upper bound: 63.3947322
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939840, upper bound: 63.3939840
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939840, upper bound: 63.3939840
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3953461, upper bound: 63.3953461
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3953461, upper bound: 63.3953461
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940622, upper bound: 63.3940622
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940622, upper bound: 63.3940622
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944410, upper bound: 63.3944410
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944410, upper bound: 63.3944410
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943821, upper bound: 63.3943821
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943821, upper bound: 63.3943821
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947128, upper bound: 63.3947128
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947128, upper bound: 63.3947128
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947665, upper bound: 63.3947665
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947665, upper bound: 63.3947665
time: 0.71 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.75 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3941265, upper bound: 63.3941265
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3941265, upper bound: 63.3941265
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3932153, upper bound: 63.3932153
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3932153, upper bound: 63.3932153
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3950662, upper bound: 63.3950662
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3950662, upper bound: 63.3950662
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949444, upper bound: 63.3949479
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949444, upper bound: 63.3950110
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3945424, upper bound: 63.3945424
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3945424, upper bound: 63.3945424
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949526, upper bound: 63.3949526
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949526, upper bound: 63.3949526
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3940519, upper bound: 63.3940519
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3940519, upper bound: 63.3940519
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3950751
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3950751, upper bound: 63.3950751
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3945944, upper bound: 63.3945944
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3945944, upper bound: 63.3945944
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3946273, upper bound: 63.3946273
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947662
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3943689, upper bound: 63.3943688
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3943689, upper bound: 63.3943689
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3942043, upper bound: 63.3942043
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3942043, upper bound: 63.3942043
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3945267, upper bound: 63.3945267
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3945267, upper bound: 63.3945267
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3945183, upper bound: 63.3945180
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3945180, upper bound: 63.3945180
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3948706, upper bound: 63.3948706
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3948706, upper bound: 63.3948706
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949036
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3931200, upper bound: 63.3931200
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3931200, upper bound: 63.3931200
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3948076, upper bound: 63.3948076
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3948076, upper bound: 63.3948076
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949004, upper bound: 63.3949004
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949004, upper bound: 63.3949004
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3945626, upper bound: 63.3945626
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3945626, upper bound: 63.3945626
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3946721, upper bound: 63.3946721
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3946721, upper bound: 63.3946721
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3956871, upper bound: 63.3956871
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3947322, upper bound: 63.3947322
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3947322, upper bound: 63.3947322
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3939840, upper bound: 63.3939840
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3939840, upper bound: 63.3939840
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3953461, upper bound: 63.3953461
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3953461, upper bound: 63.3953461
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3944477, upper bound: 63.3944477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3940622, upper bound: 63.3940622
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3940622, upper bound: 63.3940622
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3944410, upper bound: 63.3944410
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3944410, upper bound: 63.3944410
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3943821, upper bound: 63.3943821
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3943821, upper bound: 63.3943821
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3947128, upper bound: 63.3947128
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3947128, upper bound: 63.3947128
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3947665, upper bound: 63.3947665
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 0, lower bound: -63.3947665, upper bound: 63.3947665

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939471, upper bound: 63.3939471
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939471, upper bound: 63.3939471
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939471, upper bound: 63.3939471
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939471, upper bound: 63.3939471
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950662, upper bound: 63.3950662
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3950662, upper bound: 63.3950662
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949678, upper bound: 63.3949678
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949678, upper bound: 63.3949678
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943119, upper bound: 63.3943234
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943119, upper bound: 63.3943119
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949444, upper bound: 63.3949457
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949444, upper bound: 63.3950036
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941950, upper bound: 63.3941950
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3941950, upper bound: 63.3941950
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3929300, upper bound: 63.3929380
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3929300, upper bound: 63.3929394
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947888, upper bound: 63.3947888
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947888, upper bound: 63.3947888
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3934577, upper bound: 63.3934577
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3934577, upper bound: 63.3934577
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3925967, upper bound: 63.3925967
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3925967, upper bound: 63.3925967
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3934577, upper bound: 63.3934577
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3934577, upper bound: 63.3934577
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3878555, upper bound: 63.3878555
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3878555, upper bound: 63.3878554
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949264, upper bound: 63.3949264
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949264, upper bound: 63.3949264
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943119, upper bound: 63.3943119
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943119, upper bound: 63.3943119
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944051, upper bound: 63.3944051
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944051, upper bound: 63.3944051
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944595, upper bound: 63.3944595
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3944595, upper bound: 63.3944595
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945267, upper bound: 63.3945267
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945267, upper bound: 63.3945267
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946548, upper bound: 63.3946548
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946548, upper bound: 63.3946713
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947629, upper bound: 63.3947629
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943532, upper bound: 63.3943532
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943532, upper bound: 63.3943532
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943569, upper bound: 63.3943569
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943569, upper bound: 63.3943569
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942043, upper bound: 63.3942042
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942043, upper bound: 63.3942042
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942039, upper bound: 63.3942039
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942039, upper bound: 63.3942038
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942402, upper bound: 63.3942402
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942402, upper bound: 63.3942402
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943180, upper bound: 63.3943180
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943180, upper bound: 63.3943179
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945047, upper bound: 63.3945047
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945047, upper bound: 63.3945047
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945180, upper bound: 63.3945180
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945180, upper bound: 63.3945180
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948706, upper bound: 63.3948706
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948706, upper bound: 63.3948706
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947340, upper bound: 63.3947340
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947340, upper bound: 63.3947340
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945424, upper bound: 63.3945424
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945424, upper bound: 63.3945424
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943736, upper bound: 63.3943993
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943736, upper bound: 63.3943736
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942765, upper bound: 63.3942765
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942765, upper bound: 63.3942765
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945401, upper bound: 63.3945401
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945401, upper bound: 63.3945401
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949011, upper bound: 63.3949011
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943736, upper bound: 63.3943746
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943736, upper bound: 63.3943754
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948997, upper bound: 63.3948996
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3948997, upper bound: 63.3948997
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3931041, upper bound: 63.3931041
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3931041, upper bound: 63.3931041
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949352, upper bound: 63.3949352
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3862009, upper bound: 63.3862009
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3862009, upper bound: 63.3862009
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945614, upper bound: 63.3945614
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945614, upper bound: 63.3945614
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942219, upper bound: 63.3942219
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942219, upper bound: 63.3942219
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3912472, upper bound: 63.3912472
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3912472, upper bound: 63.3912472
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946721, upper bound: 63.3946721
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3946721, upper bound: 63.3946721
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3951211, upper bound: 63.3951211
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3951211, upper bound: 63.3951211
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956361, upper bound: 63.3956361
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3956361, upper bound: 63.3956361
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945800, upper bound: 63.3945800
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3945800, upper bound: 63.3945800
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3913611, upper bound: 63.3913611
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3913611, upper bound: 63.3913611
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937383, upper bound: 63.3937383
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3937383, upper bound: 63.3937383
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938223, upper bound: 63.3938223
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938223, upper bound: 63.3938223
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949444, upper bound: 63.3949444
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3949444, upper bound: 63.3949444
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3952057, upper bound: 63.3952057
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3952057, upper bound: 63.3952057
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942770, upper bound: 63.3942770
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3942770, upper bound: 63.3942770
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943821, upper bound: 63.3943821
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3943821, upper bound: 63.3943821
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940519, upper bound: 63.3940519
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940519, upper bound: 63.3940519
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940519, upper bound: 63.3940519
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3940519, upper bound: 63.3940519
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3930700, upper bound: 63.3930700
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3930700, upper bound: 63.3930700
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939060, upper bound: 63.3939060
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3939060, upper bound: 63.3939060
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3929782, upper bound: 63.3929887
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3929782, upper bound: 63.3929887
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938307, upper bound: 63.3938307
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3938307, upper bound: 63.3938307
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947128, upper bound: 63.3947128
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -63.3947128, upper bound: 63.3947128
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3933499, upper bound: 63.3933499
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -63.3933499, upper bound: 63.3933499
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.4153824, 40.2786446, -38.4153824, 40.2786446, -78.6940308, 78.6940308
1: -271.9440918, 95.2166214, -271.9440918, 95.2166214, -367.1607056, 367.1607056
2: -152.0317841, 87.0516129, -152.0317841, 87.0516129, -239.0833893, 239.0833893
3: -189.9531097, 69.9066696, -189.9531097, 69.9066696, -259.8597412, 259.8597412
4: -109.4088516, 76.1420975, -109.4088516, 76.1420975, -185.5509186, 185.5509338

Time for backsubstitution: 1.20 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.11 + 417.53 = 420.65 seconds
